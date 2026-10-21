---
layout: post
title: "Why a Memory Platform May Raise RCD VDDIO from 1.0 V to 1.2 V"
date: 2026-09-06
tags: [ddr5, memory, vddio, pmic, signal-integrity]
---

A memory-initialization workaround may look surprisingly simple: for a particular RCD or PMIC revision, program the I/O supply to `1.2 V` instead of the nominal `1.0 V`.

That change is not necessarily intended to make the device switch between two discrete modes. More often, it restores electrical and timing margin that is not available at the lower voltage. The important distinction is between **the nominal operating point** and **the point at which the interface is guaranteed to meet its specifications**.

## The short version

An RCD (registering clock driver) sits between the memory controller and the DRAM devices. It receives and retransmits clock, command, and address signals. Its output drivers must charge and discharge the package, board traces, and DRAM inputs within the allowed timing window.

Raising VDDIO can help because it gives the output stage more voltage headroom:

```text
lower VDDIO
	-> less output-voltage/current margin
	-> slower or smaller signal transitions under load
	-> less noise and timing margin
	-> training failures or intermittent command/address errors

higher VDDIO
	-> more electrical margin
	-> cleaner, faster transitions at the receiver
	-> more robust setup/hold margin
```

This is a useful engineering model, but it is not a substitute for the RCD and PMIC data sheets. The exact failure mechanism depends on the device implementation, load, frequency, voltage tolerance, and temperature.

## What "drive strength" means here

I/O drive strength is not an abstract software setting. It describes how much current an output can source or sink while still meeting its guaranteed output-voltage limits. AMD’s I/O documentation gives this relationship directly: an 8 mA driver is specified to deliver that current while maintaining the required `VOH` and `VOL` levels.

For a CMOS output, the pull-up and pull-down transistors charge or discharge the load capacitance. The resulting transition is governed by the effective resistance of the conducting path and the capacitance of the output load. A larger effective resistance or a larger load produces a slower transition.

The supply rail matters in two ways:

1. It sets the available high-level voltage and therefore the receiver’s noise margin.
2. It affects the voltage headroom available to the output transistors while they source or sink current.

If VDDIO is too low for the actual load and operating conditions, a signal may still look digital on an oscilloscope while no longer satisfying the device's guaranteed `VOH`, `VOL`, rise-time, or fall-time limits. "It still toggles" is weaker than "it meets the interface specification."

## Why timing margin is affected

The receiver does not sample an ideal square wave. It recognizes valid logic levels only after the waveform crosses its input thresholds. CMOS timing is therefore connected to both voltage and time:

- `VOH` and `VOL` describe valid output levels.
- `VIH` and `VIL` describe valid input levels.
- Propagation delay includes the time required for a loaded node to reach a valid threshold.
- Noise margin is the distance between the guaranteed output level and the receiver’s input threshold.

A weaker or slower transition consumes more of the available timing window. Depending on which edge is affected, the receiver may see the signal too late for setup or may see the old level disappear too early for hold. Voltage noise and crosstalk also have less room before they move the waveform across an input threshold.

At a high-speed memory interface, the result may not be a permanent failure. It can be an occasional training failure, a failure only at a particular frequency, or a bit error that appears only at a temperature or load corner.

## Why undervoltage does not always mean "off"

It is tempting to imagine a voltage threshold below which the chip immediately stops working. Real devices usually have a less convenient boundary. An IC can pass through several regions as its supply falls:

| Supply region | Practical meaning |
| --- | --- |
| In specification | Performance and electrical limits are guaranteed. |
| Functional but out of specification | The device may appear to work, but timing and voltage guarantees no longer apply. |
| Undefined | Behavior is unpredictable and must not be relied upon. |
| UVLO/inactive | Protection circuitry disables the device or downstream function. |

The distinction matters for debugging. A system can boot and pass a light test while operating without enough margin for its worst-case frequency, temperature, process, or load. Undervoltage-related errors may therefore look intermittent rather than catastrophic.

## Reading the 1.0 V to 1.2 V workaround

Suppose platform code contains logic equivalent to `ProgramPmicVddIoVoltage()` and selects `1.2 V` for a small set of RCD or PMIC revisions. The safest interpretation is:

- the default voltage is suitable for the normal population of parts;
- the selected revisions need additional VDDIO margin under the platform’s operating conditions;
- the workaround changes the electrical operating point, not the memory protocol;
- the change should be applied only where the platform validation and component limits allow it.

This does **not**, by itself, prove that every device of those revisions is defective or that `1.2 V` is universally safe. A revision-specific branch is evidence of a platform qualification decision. To identify the root cause precisely, correlate the branch with the component data sheet, errata, validation results, oscilloscope measurements, and the PMIC voltage limits.

The likely benefit is straightforward: the higher rail gives the RCD output drivers more voltage and current margin, helping command/address signals reach valid levels with enough time left for the DRAM receivers to sample them reliably.

## What to verify in a real failure

When investigating this kind of workaround, check the following rather than looking only at the programmed register value:

1. Measure the actual VDDIO voltage at the device during initialization and high activity. Include droop, ripple, and load-transient behavior.
2. Compare rise and fall times at the RCD output and at the DRAM receiver, not just at the power converter.
3. Check `VOH`, `VOL`, `VIH`, and `VIL` against the relevant component specifications.
4. Repeat training across frequency, temperature, DIMM population, and board or package variants.
5. Confirm that the higher setting remains within the RCD, PMIC, DRAM, and platform power limits.

The central lesson is simple: a small supply-voltage change can be a signal-integrity fix. It restores margin in the analog behavior of a digital interface, which is why the symptom may be a sporadic training or data-integrity problem rather than an immediate power failure.

## References

- [AMD: What does the drive strength of an I/O mean?](https://adaptivesupport.amd.com/s/article/38820?language=en_US) - Defines I/O drive strength in terms of source/sink current while maintaining guaranteed `VOH` and `VOL` levels.
- [MIT Computation Structures: CMOS Technology](https://computation-structures.github.io/course/lectures/L03_CMOS_Technology.html) - Explains CMOS output levels, noise margins, loaded transitions, resistance-capacitance delay, and propagation timing.
- [All About Circuits: An Explanation of Undervoltage Lockout](https://www.allaboutcircuits.com/technical-articles/an-explanation-of-undervoltage-lockout/) - Describes the distinction between specified, functional-but-out-of-specification, undefined, and inactive undervoltage regions, including possible bit errors.
