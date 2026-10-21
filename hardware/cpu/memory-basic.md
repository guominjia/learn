# DDR5 Memory Addressing: From Physical Address to Bits on the Wire

## 1. How DDR5 Capacity is Calculated

When reading the JEDEC DDR5 specification's addressing tables, entries like **2Gb x4** and **1Gb x8** sharing the same row/column configuration can be confusing. For example:

- Row Address: R0–R15 (16 bits)
- Column Address: C0–C10 (11 bits)
- Page Size: 1 KB

A common mistake is adding them all up: 16 + 11 + 10 = 37 bits → 2³⁷ = 128 GB. That is wrong.

**Page Size is derived, not an extra address dimension.** It is computed from column bits and data width:

$$\text{Page Size} = 2^{\text{Col bits}} \times \text{Data Width (bytes)} = 2^{11} \times \frac{4\text{ bits}}{8} = 1\text{ KB}$$

Adding Page Size on top of Column bits double-counts the columns. The correct capacity formula is:

$$\text{Total Capacity} = 2^{\text{Row bits}} \times 2^{\text{Col bits}} \times \text{Banks} \times \text{Data Width}$$

For **2Gb x4** with 4 Banks (2 address bits):

$$2^{16} \times 2^{11} \times 4 \times 4\text{ bits} = 2^{16+11+2+2} = 2^{31}\text{ bits} = 2\text{ Gb} \checkmark$$

The reason **2Gb x4** and **1Gb x8** share the same address table is that they use the same physical sense-amplifier array (same page size). The difference is IO width and bank count, not row/column structure.

---

## 2. x4 vs x8: Per-Die IO Width vs DIMM Width

**x4 does not mean a DIMM transfers 4 bits at a time.** The data bus width of a single DRAM die is 4 bits, but a DIMM achieves 64-bit (or 72-bit for ECC) transfers by placing multiple dies in parallel:

$$64\text{-bit DIMM} = 16 \text{ dies} \times 4\text{ bits (x4)} = 8 \text{ dies} \times 8\text{ bits (x8)}$$

Furthermore, DDR5 uses **Burst Length 16 (BL16)**, meaning each Column command streams data for 16 clock cycles:

$$64\text{ bits} \times 16 = 1024\text{ bits} = 128\text{ bytes per access}$$

The minimum DRAM transfer granularity in DDR5 is therefore **128 bytes**, larger than a CPU cache line (64 bytes). The memory controller fetches the full 128-byte burst and discards the half it does not need.

---

## 3. Bank Hierarchy Inside a Single Die

**Banks are internal structures within each die**, nothing to do with how many dies are on the DIMM.

DDR5 organizes storage as: **4 Bank Groups × 4 Banks = 16 Banks per die**.

```
Single DDR5 Die
┌──────────────────────────────┐
│  Bank Group 0                │
│  ┌────────┐  ┌────────┐      │
│  │ Bank 0 │  │ Bank 1 │      │
│  └────────┘  └────────┘      │
│  Bank Group 1                │
│  ┌────────┐  ┌────────┐      │
│  │ Bank 2 │  │ Bank 3 │      │
│  └────────┘  └────────┘      │
│  ... (16 banks total)        │
└──────────────────────────────┘
```

Banks exist to **pipeline memory latency**: while one bank is precharging (waiting tRP), the IMC can already issue an ACT to a different bank, hiding the dead cycles.

The full address hierarchy from system to storage cell:

| Level | Description |
|---|---|
| DIMM | PCB with multiple dies in parallel |
| Rank | Group of dies activated simultaneously via shared CS# |
| Die (Chip) | Individual DRAM package |
| Bank Group | Grouping inside a die (4 in DDR5) |
| Bank | Independent array within a group (4 per group) |
| Row (Page) | One activated row, loaded into sense amplifiers |
| Column | Address within the activated row |

---

## 4. A Full Memory Access: From Physical Address to Bits

Given a physical address like `0xFFF1_0000`, the Integrated Memory Controller (IMC) decodes it into the following fields and issues a sequence of DRAM commands:

```
Physical Address 0xFFF1_0000
        │
        │  IMC address decode + interleaving
        ▼
  Channel → DIMM → Rank → BG → BA → Row → Column
        │
        │  Step 1: ACT (Activate)
        │    → Send BG + BA + Row address
        │    → Entire row (1 KB page) loaded into sense amplifiers
        │    → Wait tRCD
        │
        │  Step 2: RD (Read)
        │    → Send Column address
        │    → All 16 dies output 4 bits simultaneously = 64 bits
        │    → BL16: repeat for 16 clocks = 128 bytes total
        │
        │  Step 3: PRE (Precharge)
        │    → Close the row, reset sense amplifiers
        │    → Wait tRP
        ▼
  128 bytes returned to IMC
```

Once a Rank is selected via CS#, all dies in that Rank respond in lockstep — Row and Column addresses are broadcast to them identically. The 128-byte burst aligns to 128-byte physical address boundaries at the DRAM level.

---

## 5. Ranks Are Fixed by PCB Layout

A Rank is not a software concept — it is determined at DIMM manufacturing time by how the CS# signal lines are routed on the PCB.

```
Dual-Rank x4 DIMM Example
┌──────────────────────────────────────────────┐
│  Rank 0 (16 dies)           Rank 1 (16 dies) │
│  ┌──┐┌──┐┌──┐...┌──┐       ┌──┐...┌──┐      │
│  │C0││C1││C2│...│C15│       │C16│  │C31│      │
│  └──┘└──┘└──┘...└──┘       └──┘   └──┘      │
│                                              │
│  CS0# ──── Rank 0 only      CS1# ─ Rank 1   │
│  Shared: Address / Command / Data bus        │
└──────────────────────────────────────────────┘
```

Each die's DQ lines are hard-wired to fixed positions on the data bus:

```
Rank 0, x4 dies:
  Die 0   DQ[3:0]   → bus bits [3:0]
  Die 1   DQ[3:0]   → bus bits [7:4]
  ...
  Die 15  DQ[3:0]   → bus bits [63:60]
```

This wiring is permanent. There is no dynamic reconfiguration.

**Address interleaving** is an IMC-level concern. The DRAM dies see only ACT/RD/WR commands with addresses — they have no knowledge of how the IMC mapped a physical address to them. Interleaving distributes consecutive physical addresses across Channels, Ranks, and Banks to maximize concurrency and pipeline utilization.

---

## 6. RAS: Fault Isolation Granularity

When a memory error occurs, the Machine Check Exception (MCE) hardware can report:

- Which Channel
- Which DIMM slot
- Which Rank
- The failing physical address (Row / Column / BG / BA)

**Isolating to a specific die requires knowing the DQ wiring.** Since each die owns a fixed slice of the data bus, a persistent error in bit positions [7:4] implies Die 1 is faulty — but only if you have the DIMM vendor's DQ mapping, which is not standardized.

### Chipkill Changes This

DDR5 ECC with **Chipkill (x4 SDDC)** assigns one error-correction symbol per die (4 bits = one x4 die's output). If all bits in a symbol fail, ECC can:
1. Correct the data (entire die failure tolerated)
2. **Directly identify which symbol (die) failed**

This is why x4 Chipkill is the preferred ECC mode for server DIMMs — it provides both correction capability and die-level fault identification.

### In Practice

```
MCE fires
    │
    ▼
rasdaemon / mcelog logs:
  Physical Address + Error Syndrome (which bits)
    │
    ▼
BIOS/BMC topology table needed
  (DQ swizzle map, vendor-specific)
    │
    ▼
Practical resolution: replace the entire DIMM
```

There is no field-serviceable way to replace a single DRAM die. The repair granularity in production is always **the full DIMM**.

---

## Summary

| Concept | Key Point |
|---|---|
| Page Size | Derived from Col bits × data width, not an extra address bit |
| x4 / x8 | Per-die IO width; DIMM parallelizes multiple dies for 64-bit bus |
| BL16 | DDR5 minimum transfer = 128 bytes |
| Banks | Internal die structure for latency pipelining, not die count |
| Rank | Fixed by PCB wiring; all dies in a Rank are always selected together |
| Interleaving | IMC-only; DRAM dies are unaware |
| RAS chip isolation | Requires ECC syndrome + vendor DQ map; practical fix = swap DIMM |
