---
layout: post
title: "Cache Coherency, TLB Flush, and DMA: A Practical Memory-System Tour"
date: 2026-06-24
categories: [cpu, cache, memory-management]
tags: [cache, cache-coherency, tlb, mmu, dma, snooping]
---

Modern CPUs are fast because they almost never access DRAM directly on every load or store. They hide memory latency with multiple levels of cache, translate virtual addresses through the MMU, and cache those translations in the TLB. At the same time, devices can read and write memory through DMA without asking the CPU to execute each byte transfer.

These optimizations are powerful, but they create a central problem: **there may be several cached views of the same memory at the same time**.

- Core 0 may have a cache line in its private L1 cache.
- Core 1 may have another copy of the same line.
- The TLB may remember an old virtual-to-physical mapping.
- A DMA device may update DRAM behind the CPU's back.

This post connects three related topics: cache coherency, TLB flushing, and DMA. The goal is to build a practical mental model of what the hardware and operating system must do to keep memory access correct.

## 1. Cache Is a Copy of Memory, Not Memory Itself

A CPU cache stores memory in fixed-size blocks called **cache lines**. A common cache line size is 64 bytes. When the CPU loads one byte, the cache usually fetches the entire line containing that byte.

For a single-core system, the model is already slightly complicated:

```text
CPU load/store
	│
	▼
L1 cache
	│ miss
	▼
L2 / L3 cache
	│ miss
	▼
DRAM
```

If the CPU writes to a cache line, the system must decide when DRAM is updated:

| Policy | Meaning |
|---|---|
| Write-through | Every cache write also writes the lower memory level. |
| Write-back | The cache line is marked dirty and written back later. |

Most high-performance CPUs use **write-back** caches because writing to DRAM on every store would be too slow. But write-back means that DRAM may contain an old value while a newer value exists only inside a CPU cache.

That is fine as long as the CPU cache is the only observer. It becomes hard when there are multiple observers.

## 2. Why Cache Coherency Exists

Consider two CPU cores sharing one physical address `X`:

```text
Core 0 L1 cache: X = 1
Core 1 L1 cache: X = 1
DRAM:            X = 1
```

Now Core 0 writes `X = 2` into its L1 cache. If nothing else happens, Core 1 may still read `1` from its private cache. That violates the normal programming expectation that all cores eventually agree on the value of shared memory.

**Cache coherency** is the hardware mechanism that makes caches agree on the value and ownership of cache lines.

Coherency normally works at cache-line granularity, not byte granularity. If two independent variables live in the same cache line and different cores write them, the hardware still treats the whole line as one coherence unit. This is why false sharing can destroy multicore performance.

## 3. The Core Idea: States and Ownership

Many coherence protocols are variations of MESI:

| State | Meaning |
|---|---|
| Modified | This cache has the only valid copy, and it is dirty. DRAM is stale. |
| Exclusive | This cache has the only valid copy, and it is clean. DRAM is up to date. |
| Shared | Multiple caches may have clean copies. |
| Invalid | This cache line cannot be used. |

The protocol controls legal transitions. For example:

1. Core 0 reads line `A` from memory.
2. Core 0 may hold it in `Exclusive` state.
3. Core 1 reads the same line.
4. Both cores downgrade to `Shared` state.
5. Core 0 wants to write the line.
6. Core 0 must first invalidate Core 1's copy.
7. Core 0 can then move to `Modified` state and write locally.

This allows fast local access most of the time, while still preserving a coherent view of memory.

## 4. Snoop-Based Coherency

A common implementation strategy is **snooping**. Each cache controller watches coherence traffic on an interconnect. When one core announces that it wants to read or write a cache line, other caches check whether they hold that line and respond if needed.

Example: Core 0 wants to write a line that Core 1 has cached.

```text
Core 0: "I need exclusive ownership of line A."
Core 1: snoops the request
Core 1: finds line A in Shared state
Core 1: invalidates its local copy
Core 0: receives ownership and writes line A
```

If another cache holds the line in `Modified` state, it may need to supply the latest data before the requester can proceed:

```text
Core 1 cache: line A is Modified, DRAM is stale
Core 0:       requests line A
Core 1:       snoops request and provides the dirty data
Core 0:       receives the latest line
```

Snooping is conceptually simple, but a broadcast-style snoop system becomes expensive as core count grows. Large systems often use directory-based coherency or hybrid designs to avoid broadcasting every coherence request to every cache.

## 5. Coherency Is Not the Same as Consistency

Cache coherency answers this question:

> Do all cores agree on the value of a single memory location?

Memory consistency answers a different question:

> In what order do different cores observe multiple memory operations?

For example:

```c
data = 42;
ready = 1;
```

Coherency ensures that all cores eventually agree on the values of `data` and `ready` individually. It does not, by itself, guarantee that another core sees the write to `data` before the write to `ready`. That ordering is controlled by the architecture's memory model and by barriers/fences.

So cache coherency is necessary, but not sufficient, for correct lock-free programming.

## 6. What the TLB Caches

The TLB, or **Translation Lookaside Buffer**, is a cache for address translations.

With virtual memory enabled, a CPU load starts with a virtual address. The MMU must translate it into a physical address before accessing physical memory or a physically tagged cache.

Without a TLB, each memory access might require walking page tables in DRAM:

```text
virtual address
	│
	▼
page table walk in memory
	│
	▼
physical address
	│
	▼
cache / DRAM access
```

The TLB avoids repeating that walk:

```text
virtual address
	│
	▼
TLB lookup
	│ hit
	▼
physical address
```

A TLB entry usually contains:

| Field | Purpose |
|---|---|
| Virtual page tag | Identifies the virtual page. |
| Physical page number | Gives the translated physical page. |
| Permission bits | Read, write, execute, user/supervisor. |
| Attribute bits | Cacheability, memory type, access flags, dirty flags, etc. |
| Address-space tag | Distinguishes different processes if supported. |

Because the TLB is a cache, it can become stale when the operating system changes page tables.

## 7. Why TLB Flush Is Needed

Suppose a process maps virtual page `V` to physical page `P1`:

```text
Page table: V -> P1
TLB:        V -> P1
```

Now the OS changes the mapping:

```text
Page table: V -> P2
TLB:        V -> P1   stale
```

If the CPU continues using the old TLB entry, it may access the wrong physical page. That is a serious correctness and security problem.

Therefore, after certain page-table changes, the OS must invalidate stale TLB entries. This operation is usually called a **TLB flush**, although the exact hardware operation may invalidate one page, one address space, or the entire TLB.

Common reasons for TLB invalidation include:

- Unmapping a page.
- Changing a page from writable to read-only.
- Changing executable permission.
- Changing memory attributes such as cacheability.
- Reusing a physical page for another process.
- Context switching on hardware without address-space identifiers.

On x86, writing `CR3` historically flushes non-global TLB entries for the current address space. The `INVLPG` instruction invalidates the TLB entry for a specific linear address. Newer CPUs also provide mechanisms such as PCID to reduce unnecessary flush cost.

## 8. TLB Shootdown on Multicore Systems

On a multicore system, each core may have its own TLB. Updating a page table on one core does not automatically remove stale translations from other cores.

Example:

```text
Core 0 TLB: V -> P1
Core 1 TLB: V -> P1

Core 0 changes page table: V -> P2
```

Core 0 must ensure Core 1 stops using `V -> P1`. Operating systems solve this with a **TLB shootdown**:

1. The initiating core updates the page table.
2. It sends inter-processor interrupts or architecture-specific messages to other cores using the address space.
3. Those cores invalidate the relevant TLB entries.
4. The initiating core waits until the invalidations complete.

TLB shootdowns are expensive because they coordinate multiple cores and interrupt normal execution. This is one reason memory-management-heavy workloads can scale poorly.

## 9. Cache Flush vs TLB Flush

Cache flush and TLB flush are often confused, but they solve different stale-cache problems.

| Operation | Invalidates or writes back | Protects correctness of |
|---|---|---|
| Cache maintenance | Data or instruction cache lines | Memory contents and instruction visibility |
| TLB maintenance | Address translation entries | Virtual-to-physical mappings and permissions |

Changing a page table entry may require a TLB flush. It does not necessarily require a data-cache flush.

Changing code in memory may require instruction-cache maintenance and ordering barriers. It does not necessarily require changing the TLB.

Changing memory attributes, such as mapping a region from cacheable to uncacheable, may require both cache maintenance and TLB maintenance, because both data and translation metadata are affected.

## 10. How DMA Changes the Picture

DMA means **Direct Memory Access**. A DMA-capable device can read from or write to system memory without the CPU copying every byte.

A simplified DMA transmit path looks like this:

```text
CPU prepares buffer in memory
CPU programs device/DMA engine with buffer address and length
Device reads buffer through DMA
Device sends data to external world
Device interrupts CPU when done
```

A receive path is similar:

```text
CPU allocates empty buffer
CPU gives buffer address to device
Device writes incoming data into memory through DMA
Device interrupts CPU
CPU reads completed buffer
```

DMA is essential for high-throughput IO. Without DMA, the CPU would waste huge amounts of time copying data between device registers and memory.

## 11. DMA and Cache Coherency

DMA introduces a new observer of memory: the device. If the device and CPU caches are coherent with each other, life is easier. Many modern server and desktop platforms support IO coherency, where DMA transactions participate in the coherence domain.

But not all systems are fully coherent. Embedded systems commonly require explicit cache maintenance.

Two classic problems appear.

### Problem A: Device Reads Stale Data

The CPU writes a transmit buffer, but the data remains dirty in the CPU cache and has not reached DRAM:

```text
CPU cache: buffer = new data, dirty
DRAM:      buffer = old data
Device DMA read comes from DRAM
```

If the device reads DRAM, it sees old data. Before starting DMA-to-device, software must clean/write back the cache lines that contain the buffer.

```text
CPU writes buffer
CPU cleans cache lines to memory
Device reads latest data through DMA
```

### Problem B: CPU Reads Stale Data

The device writes a receive buffer in DRAM, but the CPU already has old contents cached:

```text
CPU cache: buffer = old data
DRAM:      buffer = new data from device
CPU read hits old cache line
```

After DMA-from-device completes, software must invalidate the CPU cache lines for that buffer before the CPU reads it.

```text
Device writes buffer through DMA
CPU invalidates affected cache lines
CPU reads new data from memory
```

This is why operating systems provide DMA mapping APIs. Drivers should not casually pass arbitrary cached memory to devices and hope it works on every architecture.

## 12. DMA Addresses: Physical, Bus, and IOVA

A DMA engine does not use a process's normal virtual address. It needs an address meaningful on the IO bus.

Depending on the platform, that address may be:

| Address type | Meaning |
|---|---|
| Physical address | Actual system physical memory address. |
| Bus address | Address as seen by a device on a bus. |
| IOVA | IO virtual address translated by an IOMMU. |

An IOMMU is like an MMU for devices. It translates device-visible IO virtual addresses to physical addresses and enforces permissions. This improves isolation and allows scatter-gather mappings, but it also means DMA mappings have their own translation caches, often called IOTLBs. Those may need invalidation when DMA mappings change.

## 13. Putting It Together

Here is a practical summary:

```text
CPU data cache
	caches memory contents
	problem: stale or dirty data
	solution: coherence protocol or cache maintenance

TLB
	caches address translations
	problem: stale virtual-to-physical mapping
	solution: TLB invalidation / shootdown

DMA engine
	accesses memory without CPU load/store instructions
	problem: bypasses or participates differently in caches and translations
	solution: DMA mapping API, cache maintenance, IOMMU maintenance
```

For normal application programmers, most of this is hidden behind the OS and hardware. For kernel developers, driver writers, hypervisor developers, and performance engineers, these details matter a lot.

The key questions are always:

1. **Who can observe this memory?** CPU core, another core, device, or GPU?
2. **Which caches may contain a copy?** Data cache, instruction cache, TLB, IOTLB?
3. **What changed?** Data contents, mapping, permission, or memory type?
4. **What ordering is required?** Is a barrier needed before another observer proceeds?

Once those questions are clear, cache coherency, TLB flushing, and DMA become parts of the same larger story: keeping multiple fast views of memory synchronized enough to be correct.

## References

- [Cache Coherency](https://zhuanlan.zhihu.com/p/65245043)
- [TLB之flush操作](https://zhuanlan.zhihu.com/p/66971714)
- [Cache一致性的那些事儿 (2)--Snoop方案](https://zhuanlan.zhihu.com/p/417949142)
- [TLB的作用及工作过程](https://www.cnblogs.com/alantu2018/p/9000777.html)
- [How does a DMA controller work?](https://softwareengineering.stackexchange.com/questions/272470/how-does-a-dma-controller-work)
