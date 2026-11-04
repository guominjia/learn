---
title: "机器鸭固件学习：IMU TO DXL"
categories: [hardware]
tags: [imu]
---

## 把 `build.ps1` 翻译成 bash

`build.ps1` 做的事可以拆成三步，每一步在 Linux 上的可行性不同：

### 第一步：编译 C 源码 → `.o`（可以改）

`build.ps1` 调 `armclang.exe` 编译 7 个 `.c` 文件：

```
Core/Src/main.c, board.c, control_table.c, imu.c, protocol.c
Drivers/LSM6DSV16X/lsm6dsv16x_reg.c
Drivers/CMSIS/Device/ST/STM32G0xx/Source/system_stm32g0xx.c
```

编译参数是标准的 Arm Compiler 6（armclang）交叉编译标志：

```
--target=arm-arm-none-eabi -mcpu=cortex-m0plus -mthumb -mfloat-abi=soft
-std=c11 -Oz -g -gdwarf-4 -ffunction-sections -fdata-sections
-fshort-enums -fshort-wchar -Wall -Wextra -Wno-unused-parameter
-DSTM32G031xx -DHSE_VALUE=16000000 -D__MICROLIB
```

**armclang 本身就是 LLVM/Clang 的 Arm 交叉编译版本**，这些标志在 Linux 上的 `armclang` 里完全一样。CMSIS 头文件也支持 GCC（`cmsis_compiler.h` 里有 `#elif defined(__GNUC__)` → `#include "cmsis_gcc.h"` 分支），所以用 `arm-none-eabi-gcc` 也能编译。

### 第二步：汇编启动文件 → `.o`（需要换文件）

```
armasm --cpu=Cortex-M0plus --apcs=interwork ... startup_stm32g031xx.s
```

[startup_stm32g031xx.s](https://github.com/fanhao375/microduck-replica/tree/master/hardware/imu_to_dxl/firmware-1.0.0/Startup/startup_stm32g031xx.s#L0-L20) 是 **Arm 汇编器（armasm）语法**（KEIL/ARMCC 专用的 `.s` 文件），`armasm` 和 `gas`（GNU 汇编器）语法不兼容。ST 的 CMSIS Device 包同时提供 GNU 语法版本（`Source/Templates/gcc/startup_stm32g031xx.s`），但仓库里只收录了 armasm 版本。

### 第三步：链接 → `.axf` → `.hex`/`.bin`（需要换工具链）

```
armlink --cpu=Cortex-M0plus --library_type=microlib --entry=Reset_Handler
         --scatter=MDK-ARM/imu_to_dxl.sct ...
fromelf --i32combined ... --bin ...
```

`armlink` 用的是 Keil 的 scatter 文件（`.sct`），GNU linker `ld` 用的是 `.ld` 脚本，格式完全不同。`fromelf` 对应 `objcopy`。

---

### 实际可行的 Linux 构建方案

#### 方案 A：用 Arm Compiler 6 Linux 版（最省事）

Arm Compiler 6 有 Linux 版本（`armclang`/`armlink`/`fromelf`），命令行参数和 Windows 版完全一致。如果有 Arm Developer Suite 的 Linux 许可：

```bash
# build.sh — 直接翻译 build.ps1，工具名去掉 .exe
armclang --target=arm-arm-none-eabi -mcpu=cortex-m0plus -mthumb -mfloat-abi=soft \
  -std=c11 -Oz -g -gdwarf-4 -ffunction-sections -fdata-sections \
  -fshort-enums -fshort-wchar -Wall -Wextra -Wno-unused-parameter \
  -DSTM32G031xx -DHSE_VALUE=16000000 -D__MICROLIB \
  -I Core/Inc -I Drivers/CMSIS/Include -I Drivers/CMSIS/Device/ST/STM32G0xx/Include \
  -I Drivers/LSM6DSV16X \
  -c Core/Src/main.c -o Build/obj/main.o
# ... 其余 6 个 .c 同理
armasm --cpu=Cortex-M0plus --apcs=interwork --debug --diag_suppress=1950 \
  --pd "__MICROLIB SETA 1" -o Build/obj/startup_stm32g031xx.o Startup/startup_stm32g031xx.s
armlink --cpu=Cortex-M0plus --library_type=microlib --entry=Reset_Handler \
  --scatter MDK-ARM/imu_to_dxl.sct --map --symbols --xref \
  --info=sizes,totals,unused,stack --list Build/imu_to_dxl.map --strict \
  -o Build/imu_to_dxl.axf Build/obj/*.o
fromelf --i32combined --output Build/imu_to_dxl.hex Build/imu_to_dxl.axf
fromelf --bin --output Build/imu_to_dxl.bin Build/imu_to_dxl.axf
```

这就是 `build.ps1` 的直接翻译，几乎只改路径分隔符和去掉 `.exe`。**但需要 Arm Compiler 6 的 Linux 许可证**（Arm 商业产品）。

#### 方案 B：用 arm-none-eabi-gcc（免费，需要改 3 处）

这是大多数 Linux 用户的方案。`apt install gcc-arm-none-eabi` 即可。需要改三处：

**1. 启动文件**：从 ST 的 CMSIS Device 仓库取 GNU 语法版本

```bash
# 从 https://github.com/STMicroelectronics/cmsis-device-g0 取
# Source/Templates/gcc/startup_stm32g031xx.s
# 替换 Startup/startup_stm32g031xx.s（armasm 语法 → GNU gas 语法）
```

**2. 链接脚本**：把 Keil scatter（`.sct`）翻译成 GNU ld（`.ld`）

`.sct` 文件内容大致是：
```
LR_IROM1 0x08000000 0x00010000  {
  ER_IROM1 0x08000000 0x00010000  {
    *.o (RESET, +First)
    *(InRoot$$Sections)
    .ANY (+RO)
  }
  RW_IRAM1 0x20000000 0x00002000  {
    .ANY (+RW +ZI)
  }
}
```

对应 GNU ld：
```ld
/* imu_to_dxl.ld */
MEMORY {
  FLASH (rx)  : ORIGIN = 0x08000000, LENGTH = 64K
  RAM   (rwx) : ORIGIN = 0x20000000, LENGTH = 8K
}
ENTRY(Reset_Handler)
SECTIONS {
  .isr_vector : { KEEP(*(.isr_vector)) } > FLASH
  .text : { *(.text*) *(.rodata*) } > FLASH
  .data : { *(.data*) } > RAM AT > FLASH
  .bss : { *(.bss*) *(COMMON) } > RAM
  .stack (NOLOAD) : { . = . + 2048; } > RAM  /* 2 KiB 栈 */
}
```

**3. 编译/链接命令**：

```bash
arm-none-eabi-gcc -mcpu=cortex-m0plus -mthumb -mfloat-abi=soft \
  -std=c11 -O2 -g -ffunction-sections -fdata-sections \
  -fshort-enums -Wall -Wextra -Wno-unused-parameter \
  -DSTM32G031xx -DHSE_VALUE=16000000 \
  -I Core/Inc -I Drivers/CMSIS/Include \
  -I Drivers/CMSIS/Device/ST/STM32G0xx/Include -I Drivers/LSM6DSV16X \
  -c Core/Src/main.c -o Build/obj/main.o
# ... 其余 .c 同理（去掉 -D__MICROLIB，GCC 没有 microlib）

arm-none-eabi-gcc -mcpu=cortex-m0plus -mthumb -c \
  Startup/startup_stm32g031xx.s -o Build/obj/startup_stm32g031xx.o

arm-none-eabi-gcc -mcpu=cortex-m0plus -mthumb -nostdlib \
  -T imu_to_dxl.ld -Wl,--gc-sections -Wl,-Map=Build/imu_to_dxl.map \
  Build/obj/*.o -o Build/imu_to_dxl.elf \
  -lc -lnosys  # 或 -specs=nano.specs -specs=nosys.specs

arm-none-eabi-objcopy -O ihex Build/imu_to_dxl.elf Build/imu_to_dxl.hex
arm-none-eabi-objcopy -O binary Build/imu_to_dxl.elf Build/imu_to_dxl.bin
```

关键差异：
- `-D__MICROLIB` 去掉（microlib 是 Arm Compiler 专有的；GCC 用 newlib-nano 或 nosys）
- `--target=arm-arm-none-eabi` → `-mcpu=cortex-m0plus -mthumb`（GCC 不用 `--target`）
- `armasm` → `arm-none-eabi-gcc`（GNU gas 语法）
- `armlink --scatter` → `arm-none-eabi-gcc -T imu_to_dxl.ld`
- `fromelf` → `arm-none-eabi-objcopy`
- CMSIS 头文件已经支持 GCC（`cmsis_compiler.h` → `cmsis_gcc.h`），不需要改源码

#### 方案 C：用 CMake + arm-none-eabi-gcc（最规范）

写一个 `CMakeLists.txt` + `imu_to_dxl.ld`，用 `arm-none-eabi-gcc` + `arm-none-eabi-objcopy` 生成 `.elf`/`.hex`/`.bin`。很多 STM32 项目用 STM32CubeMX 生成的 CMake 模板就是这个路线。

---

### 烧录（Linux 上完全可行）

烧录用 OpenOCD，跨平台：

```bash
openocd -f interface/cmsis-dap.cfg -f target/stm32g0x.cfg \
  -c 'adapter speed 4000' \
  -c 'program Build/imu_to_dxl.hex verify reset exit'
```

或者用 `st-flash`（ST-Link）、`pyocd`（CMSIS-DAP/ST-Link 都支持），都是 Linux 原生的。

---

### 总结

| 步骤 | Windows (build.ps1) | Linux 方案 A (armclang) | Linux 方案 B (GCC) |
|---|---|---|---|
| 编译 .c | `armclang.exe` | `armclang`（同参数） | `arm-none-eabi-gcc`（改参数） |
| 汇编 .s | `armasm.exe` (armasm 语法) | `armasm`（同参数） | `arm-none-eabi-gcc`（换 GNU 语法启动文件） |
| 链接 | `armlink --scatter .sct` | `armlink --scatter .sct` | `arm-none-eabi-gcc -T .ld`（写 .ld） |
| 生成 hex/bin | `fromelf` | `fromelf` | `arm-none-eabi-objcopy` |
| microlib | `--library_type=microlib` | `--library_type=microlib` | 去掉，用 `-specs=nano.specs` 或 `-nostdlib` |
| 烧录 | `openocd.exe` | `openocd` | `openocd` |