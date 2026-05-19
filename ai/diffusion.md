# Understanding the DDPM Forward Diffusion Process

The **Denoising Diffusion Probabilistic Model (DDPM)** paper describes a corruption process that progressively adds small amounts of Gaussian noise to data over a series of timesteps. This post breaks down the math notation step by step so it no longer looks scary.

---

## 1. The Forward Process — One Step at a Time

Given $\mathbf{x}_{t-1}$ (the image at timestep $t{-}1$), we obtain a slightly noisier version $\mathbf{x}_t$ via:

$$
q(\mathbf{x}_t \mid \mathbf{x}_{t-1}) = \mathcal{N}\!\left(\mathbf{x}_t;\ \sqrt{1-\beta_t}\,\mathbf{x}_{t-1},\ \beta_t \mathbf{I}\right)
$$

The full forward trajectory from $\mathbf{x}_0$ to $\mathbf{x}_T$ factorizes as:

$$
q(\mathbf{x}_{1:T} \mid \mathbf{x}_0) = \prod_{t=1}^{T} q(\mathbf{x}_t \mid \mathbf{x}_{t-1})
$$

### What each symbol means

| Symbol | Meaning |
|--------|---------|
| $q(\cdot)$ | The forward (corruption) distribution — not learned, just defined |
| $\mathbf{x}_t$ | The data at timestep $t$ (a vector / tensor) |
| $\beta_t$ | The noise schedule at step $t$; a small positive scalar (e.g. 0.0001 → 0.02) |
| $\mathbf{I}$ | The identity matrix — noise is isotropic (equal in every dimension) |
| $\mathcal{N}(\mathbf{x};\boldsymbol{\mu},\boldsymbol{\Sigma})$ | A Gaussian distribution evaluated at $\mathbf{x}$ with mean $\boldsymbol{\mu}$ and covariance $\boldsymbol{\Sigma}$ |

### Equivalent sampling form

The distribution above is equivalent to:

$$
\mathbf{x}_t = \sqrt{1-\beta_t}\,\mathbf{x}_{t-1} + \sqrt{\beta_t}\,\boldsymbol{\epsilon}, \quad \boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0},\mathbf{I})
$$

Two things happen at every step:

1. **Scale down** the signal by $\sqrt{1-\beta_t}$ (slightly less than 1).
2. **Add fresh noise** scaled by $\sqrt{\beta_t}$.

---

## 2. Jumping Directly to Any Timestep

Applying the one-step formula $t$ times sequentially is expensive. Fortunately, because each step is linear-Gaussian, we can derive a **closed-form** expression to go straight from $\mathbf{x}_0$ to $\mathbf{x}_t$:

$$
q(\mathbf{x}_t \mid \mathbf{x}_0) = \mathcal{N}\!\left(\mathbf{x}_t;\ \sqrt{\bar{\alpha}_t}\,\mathbf{x}_0,\ (1-\bar{\alpha}_t)\mathbf{I}\right)
$$

where:

$$
\alpha_t = 1 - \beta_t, \qquad \bar{\alpha}_t = \prod_{i=1}^{t} \alpha_i
$$

Or equivalently:

$$
\mathbf{x}_t = \sqrt{\bar{\alpha}_t}\,\mathbf{x}_0 + \sqrt{1-\bar{\alpha}_t}\,\boldsymbol{\epsilon}, \quad \boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0},\mathbf{I})
$$

### Intuition

| Timestep | $\sqrt{\bar{\alpha}_t}$ (signal weight) | $\sqrt{1-\bar{\alpha}_t}$ (noise weight) | Result |
|----------|---------------------------------------|----------------------------------------|--------|
| Early ($t$ small) | ≈ 1 | ≈ 0 | Mostly original image |
| Late ($t$ large) | ≈ 0 | ≈ 1 | Mostly pure noise |

In diffusers, these two quantities are exposed by the scheduler as `sqrt_alpha_prod` and `sqrt_one_minus_alpha_prod`.

---

## References

- Ho, J., Jain, A., & Abbeel, P. (2020). *Denoising Diffusion Probabilistic Models*. [arXiv:2006.11239](https://arxiv.org/abs/2006.11239)
- HuggingFace Diffusers documentation: [Schedulers](https://huggingface.co/docs/diffusers/api/schedulers/overview)
