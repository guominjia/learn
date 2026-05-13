---
title: "Exposing EMIB Inter-Die Signals Using FIB Pads: A Deep Dive into Advanced Packaging Failure Analysis"
date: 2026-05-13
tags: [semiconductor, EMIB, FIB, failure-analysis, advanced-packaging, Intel]
---

# Exposing EMIB Inter-Die Signals Using FIB Pads

As chiplet-based architectures become the backbone of modern high-performance computing, the ability to debug and validate die-to-die interconnects buried deep inside a package is more critical than ever. This post explores how Intel's iSTARE and MPE Labs teams use Focused Ion Beam (FIB) techniques to expose and probe Embedded Multi-die Interconnect Bridge (EMIB) signals — a breakthrough in advanced packaging failure analysis.

---

## What Is EMIB?

**Embedded Multi-die Interconnect Bridge (EMIB)** is Intel's 2.5D packaging technology. Rather than using a full-size silicon interposer (as in traditional 2.5D designs), EMIB embeds a small, passive silicon bridge inside the package substrate to connect adjacent chiplets.

Key characteristics:

- **Heterogeneous integration**: Connects diverse dies (CPU, GPU, HBM, I/O tiles) within a single package.
- **High-density interconnects**: Signal pitch typically ranges from **36 um to 55 um** — far denser than conventional package-level routing.
- **Buried by design**: The bridge and its interconnects sit **inside** the substrate layers, invisible and physically inaccessible to external probes under normal conditions.

Products like Intel's **Ponte Vecchio** GPU leverage EMIB to stitch together dozens of tiles into a single coherent processor.

---

## The Problem: How Do You Debug What You Cannot Touch?

Once a 2.5D package is assembled, the inter-die signals traveling through the EMIB bridge are completely buried. If something goes wrong — signal integrity degradation, open circuits, shorts, or timing violations between chiplets — engineers face a fundamental challenge:

> **You cannot probe what you cannot reach.**

Traditional electrical probing assumes physical access to bond pads or test points on the package surface. EMIB signals have neither. They exist in a sealed, multi-layer sandwich of silicon and organic substrate material.

---

## The Solution: FIB-Based Signal Exposure and Pad Deposition

The collaboration between **iSTARE** and **MPE Labs** has developed a precise, multi-step technique to expose buried EMIB signals and create probeable contact points.

### Step 1 — Material Removal with Plasma-FIB (PFIB)

Accessing the buried EMIB bridge requires removing significant volumes of overlying package material (organic substrate, redistribution layers, etc.). Traditional gallium-ion FIB (Ga+ FIB) excels at nanoscale precision but is far too slow for bulk material removal at the package level.

**Plasma-FIB (PFIB)**, which uses xenon ions, provides:

- **Orders-of-magnitude faster milling rates** for large-area cross-sectioning.
- Ability to cut through thick organic and metallic layers to reach the buried silicon bridge.
- Reduced implantation damage compared to prolonged Ga+ exposure.

PFIB creates the initial access cavity, exposing the EMIB bridge surface and its interconnect traces.

### Step 2 — Precision Milling to Expose Target Signals

Once the bridge is exposed, conventional Ga+ FIB takes over for fine-scale, site-specific milling. Using CAD navigation overlays, engineers identify and isolate specific inter-die signal traces on the bridge.

This step requires:

- Sub-micron alignment accuracy to target individual traces at 36-55 um pitch.
- Careful dose control to avoid severing adjacent signals or damaging the passive silicon bridge.

### Step 3 — FIB-Induced Deposition of Conductive Pads

The exposed signal traces are too small for direct probe contact. To bridge this gap, engineers use **FIB-induced deposition (FIBID)** to grow conductive pads directly on top of the target traces.

- **Materials**: Typically tungsten (W) or platinum (Pt) deposited via gas-assisted FIB.
- **Pad size**: Scaled to be large enough for micro-probe tips (typically a few um^2) while maintaining electrical isolation from neighboring signals.
- **Electrical contact**: The deposited metal creates a low-resistance ohmic connection to the underlying trace.

### Step 4 — Electrical Probing and Measurement

With FIB pads in place, standard failure analysis instruments gain access to previously unreachable signals:

- **Oscilloscopes and time-domain reflectometry (TDR)** for signal integrity characterization.
- **Probe stations** for DC parametric measurements (continuity, leakage, resistance).
- **Real-time waveform capture** during powered operation of the device.

---

## Why Target EMIB Instead of Probing Inside the Die?

A natural question arises: *why not skip the bridge and probe the die's internal signals directly?* The answer lies in five fundamental constraints.

### 1. Physical Inaccessibility of Active Die Circuits

In flip-chip packaging, the die's **active side faces downward**, bonded to the substrate via micro-bumps. The upward-facing side is the **bulk silicon back**, often hundreds of micrometers thick. Reaching the active metal layers from the backside requires thinning through the entire silicon bulk — an extremely high-risk operation that frequently destroys the device.

### 2. Scale Mismatch: Nanometers vs. Micrometers

| Feature | Die Internal Wiring | EMIB Bridge Wiring |
|---|---|---|
| **Pitch** | Single-digit to tens of nm (e.g., Intel 4, TSMC N3) | 36-55 um |
| **FIB compatibility** | Extremely challenging; high risk of collateral damage | Well within FIB precision tolerances |

FIB's resolution is more than sufficient for EMIB-scale features. At die-internal scales, the margin for error effectively vanishes.

### 3. Signal Concentration at the Bridge

Die-internal signals are distributed across **billions of nodes** — finding a specific one is like searching for a needle in a haystack. In contrast, all cross-die communication (data buses, control signals, clocks) **converges onto the EMIB bridge**. Probing the bridge is analogous to monitoring a highway interchange rather than every street in a city.

### 4. Damage Risk and Survivability

| Target | Risk Level | Reason |
|---|---|---|
| **Active Die** | Very High | FIB ion implantation damages gate oxides, causes ESD breakdown and leakage; the die is often destroyed. |
| **EMIB Bridge** | Low | The bridge is a **passive silicon interposer** with only metal interconnects — no active transistors to damage. Minor ion implantation does not compromise functionality. |

### 5. Non-Destructive Alternatives Exist for Die-Level Analysis

The semiconductor industry already has mature, non-contact techniques for probing active die signals:

- **Electro-Optical Probing (EOP) / Electro-Optical Frequency Mapping (EOFM)**: Exploits silicon's infrared transparency to optically detect switching activity from the die backside.
- **Picosecond Imaging Circuit Analysis (PICA)**: Captures faint photon emissions from transistor switching events.
- **Laser Voltage Probing (LVP)**: Measures voltage-dependent reflectivity changes on individual transistors.

These tools handle die-level analysis without physical modification, making FIB-based probing unnecessary at that level.

---

## Why This Matters

As advanced packaging moves toward increasingly complex multi-chiplet architectures, the ability to validate and debug die-to-die interfaces is a critical enabler for:

- **Yield improvement**: Identifying systematic interconnect defects early accelerates process maturation.
- **Signal integrity validation**: Confirming that high-speed links meet electrical specifications under real operating conditions.
- **Root cause analysis**: Pinpointing whether a system failure originates in a die, the bridge, or the package substrate.
- **Design feedback**: Providing physical measurement data back to package designers and signal integrity engineers.

The iSTARE and MPE Labs technique essentially creates an **"emergency tap"** on the chip's internal data highway — extracting maximum diagnostic value with minimal physical disruption.

---

## Open Questions and Future Directions

Several interesting challenges remain in this space:

- **Active vs. passive probing**: Can FIB-pad probing be performed while the device is powered on and running real workloads, or only in static (unpowered) analysis mode?
- **Loading effects**: How much does the added capacitance of FIB pads and probe tips distort the signals being measured, especially at multi-GHz data rates?
- **Automation and throughput**: As chiplet counts increase, can PFIB/FIB workflows be automated for higher-throughput failure analysis?
- **Complementary techniques**: How does FIB-pad probing integrate with X-ray tomography, acoustic microscopy, and other non-destructive inspection methods in a comprehensive FA flow?

---

## References

1. [Site-specific metrology, inspection, and failure analysis of 3D interconnects — SPIE Digital Library](https://www.spiedigitallibrary.org/journals/JM3/volume-13/issue-01/011202/Site-specific-metrology-inspection-and-failure-analysis-of-three-dimensional/10.1117/1.JMM.13.1.011202.full)
2. [Intel Foundry Packaging Technologies](https://www.intel.com/content/www/us/en/foundry/packaging.html)
3. [EMIB Product Brief — Intel (PDF)](https://www.intel.com/content/dam/www/central-libraries/us/en/documents/2025-07/emib-product-brief.pdf)
4. [Advanced Packaging and Heterogeneous Integration — PMC / NIH](https://pmc.ncbi.nlm.nih.gov/articles/PMC8246817/)
5. [Multi-die Integration Challenges — PMC / NIH](https://pmc.ncbi.nlm.nih.gov/articles/PMC12972287/)
6. [FIB-based Failure Analysis Techniques — Springer](https://link.springer.com/article/10.1186/s42649-019-0008-2)
7. [Structural Integrity of FIB-Modified Interconnects — IEEE Xplore](https://ieeexplore.ieee.org/document/9501629/)
