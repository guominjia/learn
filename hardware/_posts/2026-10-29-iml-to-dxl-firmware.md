---
title: "机器鸭固件学习：IMU TO DXL"
categories: [hardware]
tags: [imu]
---

区分两个概念：

| 动作 | 函数 | 方向 | 说明 |
|---|---|---|---|
| **接收**字节 | `board_bus_pop` → `proto_feed` | 总线 → 协议状态机 | 从接收缓冲区取出字节，逐字节喂给协议解析器 |
| **发送**字节 | `send_packet` → `board_bus_send` | 协议状态机 → 总线 | 协议解析完一条完整指令后，组装回复包再发出去 |

## 逐行解读

```c
while (board_bus_pop(&byte, &received_us)) {   // 从接收 FIFO 里逐字节取出
    proto_feed(&protocol, byte, received_us);   // 把这个字节喂给协议状态机
}
```

- `board_bus_pop`：从 UART 接收环形缓冲区里**弹出一个字节**（中断里收到的、暂存在缓冲区的）。`received_us` 是这个字节到达的时间戳。
- `proto_feed`：把字节交给协议状态机（飞特 SCS/STS / Dynamixel 协议解析）。状态机内部逐字节匹配帧头、长度、校验等，**只有当一帧完整接收完毕**后才会触发指令处理。

## 发送发生在哪里？

发送不在 `proto_feed` 里逐字节触发，而是在协议状态机**解析出一条完整指令**后，通过回调 `send_packet` 一次性把整个回复包发出去：

```c
static void send_packet(void *user, const uint8_t *data, uint16_t length) {
    proto_context *context = (proto_context *)user;
    if (board_bus_send(data, length) != 0)        // 整包发送
        ++context->cancelled_replies;
}
```

这个回调在 `proto_init` 时注册：

```c
proto_init(&protocol, control_table_read, send_packet, &protocol);
```

## 整体流程

```
UART RX 中断 → board_bus 缓冲区（FIFO）
                    ↓
main loop: board_bus_pop 逐字节取出
                    ↓
          proto_feed 逐字节喂给状态机
                    ↓
          一帧完整 → 状态机处理指令
                    ↓
          send_packet → board_bus_send 整包回复
```

所以这段 `while` 循环的作用是**排空接收缓冲区**，把积压的字节尽快交给协议解析器。真正的总线发送是协议层在帧完整后一次性完成的。

## `proto_poll` 做什么

`proto_poll` **不是"每毫秒拉一次"**，它是主循环每次迭代都调用的**时间驱动状态机推进器**。它做三件和时间相关的事：

### 1. 丢弃超时的半截帧（字节间超时）

```c
if (ctx->rx_length != 0u &&
    (uint32_t)(now_us - ctx->last_byte_us) > PROTO_BYTE_TIMEOUT_US) {
    ctx->rx_length = 0u;
    ++ctx->receive_timeouts;
}
```

如果一帧收到一半（`rx_length != 0`）但距离最后一个字节已超过 `PROTO_BYTE_TIMEOUT_US`（500µs），说明发送方中断了或丢字节了，直接丢弃这个半截帧。`proto_feed` 里也有同样的检查，但 `proto_poll` 覆盖了"字节之间没有新数据进来"的情况——`proto_feed` 只在收到新字节时才检查，如果总线安静了，只有 `proto_poll` 能触发超时丢弃。

### 2. Sync Read 排队等轮到 → 跳过离线设备

```c
if (ctx->pending == PROTO_WAIT_TURN) {
    if ((uint32_t)(now_us - ctx->transaction_start_us) >= PROTO_TRANSACTION_TIMEOUT_US) {
        cancel_reply(ctx);    // 整轮超时（25ms），放弃
        return;
    }
    if (ctx->rx_length == 0u &&
        (uint32_t)(now_us - ctx->quiet_since_us) >= PROTO_SLOT_GAP_US) {
        ++ctx->skipped_slots;
        ++ctx->slot;
        ctx->quiet_since_us = now_us;
        if (ctx->slot >= ctx->turn) {
            ctx->pending = PROTO_READY;    // 轮到我了
            ctx->reply_due_us = now_us;
        }
    }
}
```

Sync Read（广播 0x82）时，总线上多个设备按 ID 列表顺序依次回复。如果排在我前面的设备静默超过 `PROTO_SLOT_GAP_US`（1ms），`proto_poll` 就判定它离线，跳过它（`++slot`），直到轮到我（`slot >= turn`）就标记 `PROTO_READY` 准备发送。整个事务最多等 `PROTO_TRANSACTION_TIMEOUT_US`（25ms），超时就放弃。

### 3. 到点了就发送回复

```c
if (ctx->pending == PROTO_READY && (int32_t)(now_us - ctx->reply_due_us) >= 0) {
    if ((uint32_t)(now_us - ctx->reply_due_us) > PROTO_REPLY_EXPIRY_US ||
        ctx->rx_length != 0u) {
        cancel_reply(ctx);    // 太晚了（>2ms）或有人开始说话了，放弃
        return;
    }
    ctx->pending = PROTO_IDLE;
    ctx->tx_length = 0u;
    if (ctx->send != 0) {
        ctx->send(ctx->user, ctx->tx, ctx->tx_length);  // 真正发送！
    }
    ++ctx->tx_packets;
}
```

当 `pending == PROTO_READY` 且当前时间 ≥ `reply_due_us`（即 `T_resp = 50µs` 延迟已到），就调用 `send` 回调（即 `main.c` 里的 `send_packet` → `board_bus_send`）把回复包发出去。但如果已经过了 `PROTO_REPLY_EXPIRY_US`（2ms）或总线又有人开始说话了（`rx_length != 0`），就取消回复避免冲突。

## 总结

| `proto_poll` 的职责 | 触发条件 | 时间常量 |
|---|---|---|
| 丢弃半截帧 | 字节间隔超时 | `PROTO_BYTE_TIMEOUT_US` = 500µs |
| 跳过离线设备（Sync Read 排队） | 总线安静超过 slot gap | `PROTO_SLOT_GAP_US` = 1ms |
| 放弃整轮 Sync Read | 事务总超时 | `PROTO_TRANSACTION_TIMEOUT_US` = 25ms |
| **发送回复包** | `reply_due_us` 到了 | `PROTO_RESPONSE_DELAY_US` = 50µs |
| 取消过期回复 | 到点太晚 | `PROTO_REPLY_EXPIRY_US` = 2ms |

关键点：**回复不是在 `proto_feed` 收完帧时立刻发的**。`proto_feed`/`execute` 只是把回复帧组好放进 `tx[]` 并标记 `pending = PROTO_READY` + `reply_due_us = now + 50µs`。真正调用 `board_bus_send` 把字节推上总线是在 `proto_poll` 里，等 50µs 响应延迟到了才发。这就是为什么主循环要高频调用 `proto_poll`——头文件注释说目标 <20µs 一次，因为 `T_resp` 只有 50µs，轮询太慢就会错过发送窗口。

`proto_feed` 里也有一行关键逻辑：如果收到新字节时发现 `pending == PROTO_READY`（正等着发回复但还没到点），会立刻 `cancel_reply`——别人开始说话了，我不能抢总线。所以"排空 RX 再 poll"的顺序很重要。