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

## `scan` 做什么

```
static void scan(proto_context *ctx, uint32_t now_us)
```

这是 **Dynamixel Protocol 1.0** 风格的流式帧解析器：从接收缓冲区 `rx` 里重新同步、找帧头、校验，再执行完整帧。

## 帧格式

```
idx:  0    1    2    3    4      5 .. N+4   LEN+3
      FF   FF   ID   LEN  INSTR  PARAM...   CHK
LEN   = 参数个数 + 2 (INSTR + CHK)
total = LEN + 4
CHK   = ~(ID + LEN + INSTR + PARAMs) & 0xFF
```

## 逐步分析

| 步骤 | 代码 | 作用 |
|---|---|---|
| 1. 找同步头 | `while (skip < len && !(rx[skip]==FF && (末字节 \|\| rx[skip+1]==FF)))` | 丢掉 `FF FF` 前面的垃圾。末尾单个 `FF` 会被**保留**，因为它可能是下一个帧头的前半部分 |
| 2. 等头部 | `rx_length < 4` → return | 需要读到 `FF FF ID LEN` 才能继续 |
| 3. 多余前导 | `rx[2]==FF` → drop 1 | `FF FF FF ...`：ID 0xFF 在 1.0 协议中不合法（广播是 0xFE），所以按多出来的前导字节处理，逐字节滑动 |
| 4. 长度合法性 | `LEN<2` 或 `LEN+4 > CAPACITY` → malformed，drop 1 | LEN 至少包含 INSTR+CHK；同时防止超出缓冲区 |
| 5. 等整帧 | `rx_length < total` → return | 帧还没收完整 |
| 6. 校验 | `checksum(rx+2, total-3)` vs `rx[total-1]` | 覆盖 ID、LEN、INSTR、PARAMs，正好 `LEN+1 = total-3` 字节，**范围正确** |
| 7. 成功 | `execute` 后 `drop_front(total)` | 消费整帧，继续 `for(;;)` 处理缓冲区中的下一帧 |

## 设计上做对的地方

- **出错只丢 1 字节**（不是丢 `total`）。假帧头可能出现在数据中间，丢 1 字节后重新扫描，可以找回紧跟着的真实帧，不会把它一起丢掉。
- **循环一定会结束**：每个 `continue` 分支至少丢 1 字节，成功分支丢 `total` 字节，其他情况都 `return`。
- 一次调用可以处理多帧，适合在 ISR 之后的主循环里调用。
- 类型提升没有问题：`rx[3] + 4u` 的结果最大为 259，转成 `uint16_t` 不会截断。

## 潜在问题和建议

1. **假 LEN 导致停顿（最主要的问题）**
   噪声可能构造出 `FF FF xx C8`（LEN=200）。这时第 5 步会一直等 204 字节。在半双工主从总线上，主机发完一帧会等应答，总线可能空闲，所以从机会一直卡住，直到主机多次重试、凑够字节、校验失败后才能恢复。
   `now_us` 已经传进来但没有用在解析上。建议加**字节间超时**：超时后 `drop_front(ctx, 1)` 并重新扫描，或者直接清空缓冲区。
   ```c
   if (ctx->rx_length < total) {
       if ((uint32_t)(now_us - ctx->last_rx_us) > PROTO_RX_TIMEOUT_US) {
           drop_front(ctx, 1u);
           continue;
       }
       return;
   }
   ```

2. **`drop_front` 的开销**
   如果 `drop_front` 用 `memmove` 实现，遇到垃圾数据时会反复丢 1 字节，最坏是 $O(n^2)$。缓冲区约 260 字节时可以接受；缓冲区更大时，可以考虑环形缓冲区或读索引。

3. **缓冲区容量约束**
   必须保证 `rx` 的实际容量 ≥ `PROTO_PACKET_CAPACITY`，否则第 5 步会永远等不到整帧。另外，如果 `PROTO_PACKET_CAPACITY ≥ 259`，第 4 步的上限检查永远不会触发，这本身没问题。

4. **`execute` 的约束**
   `execute` 直接读 `ctx->rx`，之后才 `drop_front`，所以它**不能**修改 `rx` / `rx_length`，也不能重入 `scan`，例如在回调里收到回显数据时。另外，ID 过滤（只响应本机 ID 或 0xFE）需要在 `execute` 里做，因为这里会接受任意 ID（0xFF 除外）。

5. **统计计数会偏高**
   在噪声中，每次假同步都会让 `malformed_packets` / `crc_errors` 加 1，所以这两个计数反映的是"假同步次数"，不是"真实坏帧数"。如果只用来诊断，可以接受。

6. **不兼容 Protocol 2.0**
   2.0 的帧头 `FF FF FD 00` 会被当成 ID=0xFD、LEN=0，计入 malformed 后被跳过。如果总线上两种协议共存，需要单独处理。

**结论**：同步、校验范围和错误恢复都正确。主要缺陷是**没有接收超时**，假 LEN 会让解析器停顿；建议利用已有的 `now_us` 加上字节间超时。