# Trusted Computation in the Real World: TPM, DICE/DPE, and the First-Trust Problem

Trusted computation is often presented as a neat cryptographic story: measure software, sign evidence, verify remotely, done. In practice, the hard part is not the math. The hard part is deciding **what to trust first**, and how to keep costs manageable over time.

This post summarizes a practical mental model for TPM-based attestation, DICE/DPE on devices without TPMs, and what all of this means for individual users versus large organizations.

---

## 1) What problem are we actually solving?

Remote attestation tries to answer two questions:

1. **Who is this device/workload?**
2. **What software state is it in right now?**

The goal is not to prove absolute security. The goal is to provide cryptographic evidence for identity and integrity so a verifier can apply policy.

---

## 2) TPM model: PCR extend + quote

In a TPM flow, boot components are measured and extended into PCRs:
$$
PCR_{new} = H(PCR_{old} \|\| measurement)
$$
Later, the platform produces a signed quote over PCR values (plus a nonce) using an attestation key.

### Key insight

- A verifier does **not** blindly trust the vendor.
- The verifier checks a chain of evidence:
	- key/certificate chain,
	- PCR values against policy/baseline,
	- nonce freshness (anti-replay).

So trust is policy-driven by the relying party, not merely a vendor statement.

---

## 3) No TPM? DICE replaces PCR state with key-derivation state

DICE (Device Identifier Composition Engine) is designed for constrained systems (including those without TPM).

Instead of storing a hash chain in PCRs, DICE builds a *derived identity chain*:

$$
CDI_{next} = KDF(CDI_{prev}, measurement, context)
$$

Where CDI is Compound Device Identifier material derived layer by layer from a hardware root secret (for example, UDS in immutable hardware).

If any measured layer changes, downstream CDI/key material changes too, so old identities no longer match the new state.

### Where DPE fits

- **DICE** primarily establishes identity across measured layers.
- **DPE (DICE Protection Environment)** operationalizes this at runtime:
	- context creation/isolation,
	- per-context derivation,
	- attestation evidence production,
	- lifecycle control.

Think of DPE as a runtime trust-context manager built on top of DICE-style roots.

---

## 4) “Do we need to re-verify every step every time?”

No. That would be too expensive.

Real systems use a staged model:

- Build trust chain at boot (or when context is created).
- Reuse cached intermediate evidence where policy allows.
- Re-attest selectively on high-risk operations, updates, role changes, or time-based freshness windows.

So both TPM and DICE ecosystems converge on the same operations principle: **establish once, prove on demand**.

---

## 5) The first-trust problem (bootstrapping)

A common misunderstanding is: “If I don’t already know PCR values, first attestation is meaningless.”

The deeper truth:

- You can never derive trust from zero assumptions.
- Every system has a trust anchor assumption.

What you can do is reduce and audit that assumption:

- controlled first enrollment,
- signed firmware/image provenance,
- inventory binding (device identity + first known-good evidence),
- least-privilege initial access,
- continuous revalidation.

Cryptography makes later verification strong and repeatable. It does not remove governance and supply-chain trust decisions.

---

## 6) “Is this only for governments and big companies?”

Enterprise-grade attestation pipelines are resource-intensive, yes. But individuals can still get meaningful benefits with lightweight controls:

- enable Secure Boot,
- enable disk encryption with hardware protection (TPM/LUKS integration),
- keep firmware and OS updated,
- install signed packages from trusted channels,
- verify checksums/signatures for sensitive software,
- segment risky workloads (VM/container/network boundaries).

You may not get a full remote-attestation control plane, but you can significantly raise tampering cost.

---

## 7) “What if I do not trust the vendor at all?”

Then shift from *vendor trust* to *verifiability and containment*:

- prefer open-source + auditable supply chains,
- use reproducible-build ecosystems when possible,
- isolate untrusted workloads and minimize data exposure,
- enforce outbound network controls and logging,
- keep cryptographic keys under your own control.

The strategy is zero trust in practice: assume compromise is possible, design to limit blast radius.

---

## 8) Reproducible builds: Debian, Ubuntu, CentOS

At a high level:

- **Debian**: strongest public reproducible-build maturity among the three, with long-running dashboards and measurable coverage (high but not 100%).
- **Ubuntu**: has ongoing work and inherits from Debian ecosystem patterns, but reproducibility visibility/coverage is generally less explicit at full-repo level.
- **CentOS/Stream**: not typically positioned as a reproducible-build reference ecosystem in the same way Debian is.

For users prioritizing independent verification, Debian currently offers the clearest path among these options.

---

## 9) Practical conclusion

Trusted computation is not “vendor says so.” It is:

1. a cryptographic evidence system (TPM or DICE/DPE),
2. a verifier policy system,
3. a trust-governance process for bootstrap and lifecycle.

If you remember one sentence, use this:

> Cryptography can prove consistency with assumptions; it cannot eliminate the need to choose assumptions.

---

## Related Links / References

- DICE: <https://trustedcomputinggroup.org/what-is-a-device-identifier-composition-engine-dice/>
  - <https://github.com/veraison/dice>
- TCG DICE Work Group: <https://trustedcomputinggroup.org/work-groups/dice-architectures/>
- DICE Attestation Architecture (TCG): <https://trustedcomputinggroup.org/resource/dice-attestation-architecture/>
- DICE Protection Environment (TCG): <https://trustedcomputinggroup.org/resource/dice-protection-environment/>
- DICE Protection Environment Command Response Buffer (DPECRB): <https://trustedcomputinggroup.org/resource/dpecrb/>
- Reproducible Builds (project): <https://reproducible-builds.org/>
- Reproducible Builds — Involved Projects: <https://reproducible-builds.org/who/projects/>
- Debian Reproducible Builds Wiki: <https://wiki.debian.org/ReproducibleBuilds>
- Debian reproducibility dashboard: <https://tests.reproducible-builds.org/debian/>

