# Proofs

This page contains the proofs of the theoretical results stated in Section 2.4 of the [README](README.md#24-optimizers).

## 1. Implicit Bias of AdamW

**Theorem.** (*Xie and Li 2024, Theorem 1.1*) For any continuously differentiable objective function $`L: \mathbb{R}^d \to \mathbb{R}`$, $`\beta_1\leq \beta_2 \lt 1`$, initialization $`x_0`$ and non-increasing learning rate $`\{\eta_t\}_{t=1}^\infty`$ such that $`\sum_{t}\eta_t = \infty`$, if the iterates of AdamW on $`L`$ converge to some $`x_\infty`$, then $`x_\infty`$ is a KKT point of the constrained optimization problem

```math
\min_{||x||_\infty \leq 1/\lambda} L(x).
```

If $`L`$ is additionally convex, then AdamW converges to the constrained minimizer,

```math
x_\infty \in \text{argmin}_{||x||_\infty \leq 1/\lambda} L(x).
```

*Proof (sketch).* We present a proof sketch that only proves the theorem of interest, omitting the interesting connections with Frank-Wolfe and SignGD. Interested readers are encouraged to read [Xie and Li (2024)](https://arxiv.org/abs/2404.04454). We begin by recalling the definition of a KKT point. Consider the general continuous optimization problem

```math
\begin{aligned}
\min_{x\in \mathbb{R}^n} &~~ f(x) \\
\text{subject to} &~~ g_i(x) \leq 0,\quad i =1,\dots, m \\
&~~ h_j(x) = 0,\quad j =1,\dots, p
\end{aligned}
```

where $`f, g_i, h_i: \mathbb{R}^n \to \mathbb{R}`$ are all continuously-differentiable functions. Define the Lagrangian
```math
\mathcal{L}(x, \mu, \nu) = f(x) + \langle \mu, g(x)\rangle + \langle \nu, h(x)\rangle.
```

The vectors $`\mu, \nu`$ are generalized Lagrange multiplers called *dual variables*.

A point $`(x^{\ast}, \mu^{\ast}, \nu^{\ast}) \in \mathbb{R}^n\times \mathbb{R}^m \times \mathbb{R}^p`$ is a *Karush-Kuhn-Tucker (KKT)* point if it satisfies the following conditions:

1. *Stationarity*: $`0 \in \partial \mathcal{L}(x^{\ast}, \mu^{\ast}, \nu^{\ast})`$
2. *Primal Feasibility*: $`g_i(x^{\ast}) \leq 0`$ and $`h_j(x^{\ast}) = 0`$ for all $`i,j`$
3. *Dual Feasibility*: $`\mu_i^{\ast} \geq 0`$ for all $`i=1,\dots, m`$
4. *Complementary Slackness*: $`\langle \mu^{\ast}, g(x^{\ast})\rangle =0`$.

A KKT point with positive-definite Hessian is a local minimum. Moreover, if the objective function $`f`$ and the $`g_i`$ are convex, and $`h_i`$ are affine, then $`(x^{\ast}, \lambda^{\ast}, \nu^{\ast})`$ being a KKT point implies that $`x^{\ast}`$ is a global minimum of $`f`$.

However, not every local minimum $`x^{\ast}`$ is a KKT point. In order for it to be so, the point $`x^{\ast}`$ must also satisfy a regularity condition; one example is Linear Independence Constraint Qualification (LICQ), which states that the gradients of the $`h_i`$ and the *active* $`g_i`$ are linearly-independent at $`x^{\ast}`$.

It suffices to prove the first part of the theorem. Consider the Lagrangian of the constrained optimization problem:

<div id="eq:optlag">

```math
\mathcal{L}(x, \mu) = L(x) + \mu\bigg(||x||_\infty - \frac{1}{\lambda}\bigg). \tag{3}
```

</div>

Xie and Li proved Theorem 1.1 by establishing two things:

1. **Lemma 3.8** *(Xie and Li 2024).* $`x`$ is a KKT point of $`\min_{||x||_\infty \leq 1/\lambda} L(x)`$ if and only if 

    <div id="eq:kktchar">

    ```math
    ||x||_\infty \leq \frac{1}{\lambda}\quad\text{and}\quad\langle -\lambda x, \nabla L(x)\rangle = ||\nabla L(x)||_1.\tag{4}
    ```

    </div>

2. The limit point of AdamW satisfies the condition of Lemma 3.8.

We begin by a proof of Lemma 3.8.

*Proof. (Lemma 3.8)* Primal feasibility for constrained optimization problem [3](#eq:optlag) says $`||x^{\ast}||_\infty \leq 1/\lambda`$ while stationarity says

```math
0\in \partial\mathcal{L}(x^{\ast}, \mu^{\ast}) = \nabla L(x^{\ast}) + \mu \partial ||x^{\ast}||_\infty,
```

or $`-\nabla L(x^{\ast}) \in \mu \partial ||x^{\ast}||_\infty`$. Since $`\mu\partial ||x^{\ast}||_\infty = \{ w\in \mathbb{R}^d \mid \langle w, x^{\ast}\rangle = \mu||x^{\ast}||_\infty \text{ and }||w||_1 \leq \mu \}`$, stationarity implies

```math
\langle -\nabla L(x^{\ast}), x^{\ast} \rangle = \mu||x^{\ast}||_\infty\text{ and }||-\nabla L(x^{\ast})||_1 \leq \mu.
```

Expand on the cases of primal feasibility:

- $`x^{\ast}`$ is an interior point ($`||x^{\ast}||_\infty \lt 1/\lambda`$): by complementary slackness, $`\mu^{\ast} = 0`$, which implies $`\nabla L(x^{\ast}) = 0`$.
- $`x^{\ast}`$ is a boundary point ($`||x^{\ast}||_\infty = 1/\lambda`$): the minimal value of $`\mu`$ required to satisfy stationarity is $`\mu = ||\nabla L(x^{\ast})||_1`$.

In both cases, it's true that for a KKT point $`(x^{\ast}, \mu^{\ast})`$:
```math
\langle -\nabla L(x^{\ast}), x^{\ast} \rangle = \mu^{\ast}||x^{\ast}||_\infty = \frac{1}{\lambda}|| \nabla L(x^{\ast})||_1,\quad\implies\quad \langle -\lambda x^{\ast}, \nabla L(x^{\ast})\rangle = ||\nabla L(x^{\ast})||_1.
```

The reverse direction follows by choosing $`\mu^{\ast} = ||\nabla L(x^{\ast})||_1`$. Assume $`||x^{\ast}||_\infty\leq 1/\lambda`$ and $`\langle -\lambda x^{\ast}, \nabla L(x^{\ast})\rangle = ||\nabla L(x^{\ast})||_1`$. If $`\nabla L(x^{\ast}) = 0`$ then all four KKT conditions are vacuously true. Suppose then that $`\nabla L(x^{\ast})\neq 0`$. Primal and dual feasibility are manifestly satisfied. To verify complementary slackness, note that by Hölder's inequality

```math
||\nabla L(x^{\ast})||_1 = \langle -\lambda x^{\ast}, \nabla L(x^{\ast})\rangle \leq ||\lambda x^{\ast}||_\infty\cdot ||\nabla L(x^{\ast})||_1 \leq ||\nabla L(x^{\ast})||_1.
```

Therefore both inequalities are equalities. Dividing through by $`||\nabla L(x^{\ast})||_1 \gt 0`$ gives $`||\lambda x^{\ast}||_\infty = 1`$, so

```math
\mu^{\ast} \left( ||x^{\ast}||_\infty - \frac{1}{\lambda}\right) = 0.
```

To prove stationarity note that $`0 \in \nabla L(x^{\ast}) + \mu^{\ast} \partial ||x^{\ast}||_\infty`$ is equivalent to $`-\frac{1}{\mu^{\ast}} \nabla L(x^{\ast}) \in \partial ||x^{\ast}||_\infty`$. By properties of dual norm subgradients, this requires

```math
|| - \frac{1}{\mu^{\ast}} \nabla L(x^{\ast})||_1 = 1\quad \text{and} \quad \left\langle -\frac{1}{\mu^{\ast}} \nabla L(x^{\ast}), x^{\ast} \right\rangle = ||x^{\ast}||_\infty.
```

The first condition is true by substitution of $`\mu^{\ast}`$. The second condition is true since

```math
\left\langle -\frac{1}{\mu^{\ast}} \nabla L(x^{\ast}), x^{\ast} \right\rangle = \frac{\langle -\lambda x^{\ast}, \nabla L(x^{\ast})\rangle}{\lambda \mu^{\ast}} = \frac{||\nabla L(x^{\ast})||_1}{\lambda\mu^{\ast}} = \frac{1}{\lambda} = ||x^{\ast}||_\infty. \qquad\Box
```

We now finish the proof sketch of Theorem 1.1 by proving that the limit point $`x_\infty`$ of AdamW satisfies condition [4](#eq:kktchar). The following lemma will be of use

**Lemma** *(Weighted Cesàro).* If a sequence $`\{a_t\}_{t=0}^\infty`$ on a normed space converges to $`a_t\to a`$, and another sequence $`\{\eta_t\}_{t=0}^\infty`$ has $`\eta_t \geq 0`$ and $`S_T = \sum_{t\leq T}\eta_t`$ has $`\lim_{T\to\infty} S_T = \infty`$, then

```math
\frac{1}{S_T} \sum_{t\leq T} \eta_t a_t \to a.
```

*Proof.* Given $`\epsilon \gt 0`$, pick $`t'`$ such that $`||a_t - a||\leq \epsilon`$ for all $`t \gt t'`$. Then by the triangle inequality,

```math
\left|\left|\frac{1}{S_T} \sum_{t\leq T} \eta_t(a_t - a)\right|\right| \leq \frac{1}{S_T}\sum_{t\leq T} \eta_t ||a_t - a|| \leq \frac{1}{S_T}\sum_{t\leq t'} \eta_t ||a_t - a|| + \epsilon.
```

Taking the limit as $`T\to\infty`$, the first term on the RHS is a fixed number divided by a quantity which goes to zero. The Lemma follows by rearrangement of the LHS.$`\qquad\Box`$

Proceeding to the main proof, let

```math
\begin{aligned}
g_t &= \nabla L(x_{t-1})\\
m_t &= \beta_1 m_{t-1} + (1-\beta_1)g_t\\
v_t &= \beta_2 v_{t-1} + (1-\beta_2)g_t^2\\
\Delta_t &= \frac{m_t}{\sqrt{v_t}}\quad\text{coordinate-wise}\\
x_t &= (1-\lambda\eta_t)x_{t-1} - \eta_t \Delta_t
\end{aligned}
```

with $`m_0 = v_0 = 0`$. Additionally, define $`S_{t} = \sum_{t'\leq t}\eta_{t'}`$ and the *AdamW average update size*

```math
\overline{\Delta}_t = \frac{1}{S_t} \sum_{t' \leq t} \eta_{t'}\Delta_{t'}.
```

Assume that AdamW converges as $`t\to\infty`$, so $`x_t\to x_\infty`$, $`g_t \to g_\infty`$, and that $`S_t \to \infty`$.

Xie and Li's proof relies on finding an upper bound of $`\overline{\Delta}_t`$ as $`t\to\infty`$, despite the fact that individual updates $`\Delta_t`$ can have unbounded magnitude depending on the choice of $`\beta_1, \beta_2`$, and that will be our goal as well.

The AdamW update rule implies $`\eta_t\Delta_t = (x_{t-1} - x_t) - \lambda \eta_t x_{t-1}`$. Summing up to $`T`$ and dividing by $`S_T`$,

```math
\overline{\Delta}_T = \frac{1}{S_T} \sum_{t\leq T} \left[ (x_{t-1} - x_t) - \lambda \eta_t x_{t-1}\right] = \frac{1}{S_T}\left[ (x_0 - x_T) - \lambda \sum_{t\leq T} \eta_t x_{t-1}\right].
```

Taking the limit as $`T\to\infty`$ and using Weighted Cesàro on the second term, we find $`\overline{\Delta}_\infty`$ is well-defined and given by $`-\lambda x_\infty`$.

Next we turn to analyzing the components $`g_{t, i}`$ of the gradient as $`t\to\infty`$. Suppose $`g_{t, i} \to g_{\infty, i}\neq 0`$. Since the EMA of a convergent sequence converges to its limit, $`m_{t, i} \to g_{\infty, i}`$, $`v_{t, i} \to g_{\infty, i}^2`$. It follows that $`\Delta_{t, i} \to \text{sign}(g_{\infty, i})`$. By Weighted Cesàro it follows that $`|\overline{\Delta_{t, i}}| \to 1`$.

The trickier case is if $`g_{\infty, i} = 0`$. In this case, Xie and Li prove that $`\overline{\Delta}_{t, i} \leq 1`$ as $`t\to\infty`$ directly. Combined with the case when $`g_{\infty, i} \neq 0`$, this implies $`||\overline{\Delta}_\infty||_\infty\leq 1`$ and as a result $`||x_\infty||_\infty \leq 1/\lambda`$. Moreover,

```math
\langle g_\infty, \overline{\Delta}_\infty \rangle = \sum_{i\mid g_{\infty, i}\neq 0} g_{\infty, i} \text{ sign}(g_{\infty, i}) = ||g_\infty||_1.
```

Substituting $`\overline{\Delta}_\infty = -\lambda x_\infty`$, we see that the limit point $`x_\infty`$ satisfies the conditions of Lemma 3.8 and thus is a KKT point of $`\min_{||x||_\infty \leq 1/\lambda} L(x)`$.

It remains to prove that $`\overline{\Delta}_{t, i}\leq 1`$ as $`t\to\infty`$ in the case that $`g_{\infty, i} \to 0`$. To simplify notation we fix an index $`i`$ and suppress it. It is useful to define $`L_s = \log(v_s /v_1)`$ for $`s\geq 1`$. Then $`L_s \geq (s-1)\log\beta_2`$ since $`v_s \geq \beta_2 v_{s-1}`$, and there exists $`s_0`$ such that $`L_s \leq 0`$ for all $`s\geq s_0`$ since $`v_s \to 0`$ asymptotically. We begin by proving the following Lemma.

**Lemma.** For non-increasing learning schedule $`\eta_t`$ and every $`j\geq 1`$, $`T \geq j + 1`$,

```math
\sum_{t = j+1}^T \eta_t(L_t - L_{t-j}) \leq C_j := \eta_1\left( s_0 \max_{s \lt s_0} |L_s| + \frac{j^2}{2}|\log \beta_2| \right).
```

*Proof.* We proceed in two cases: (1) if $`T\geq 2j`$ and (2) if $`j+1\leq T \lt 2j`$. Let $`s = t-j`$.

For case 1, we can write the LHS as
```math
\sum_{t = j+1}^T \eta_t L_t - \sum_{s=1}^{T-j} \eta_{s+j}L_s = \sum_{s=j+1}^{T-j}(\eta_s - \eta_{s+j}) L_s + \sum_{s=T-j+1}^T \eta_s L_s - \sum_{s=1}^j \eta_{s+j}L_s
```

Contributions to the first two terms run over disjoint indices $`s`$ and moreover $`L_s \gt 0`$ only if $`s \lt s_0`$. Since $`0\leq \eta_s - \eta_{s+j} \leq \eta_s \leq \eta_1`$, their sum is therefore bounded by $`\eta_1 s_0 \max_{s \lt s_0} |L_s|`$. The third term is bounded above by

```math
- \sum_{s=1}^j \eta_{s+j}L_s \leq \eta_1 |\log\beta_2| \sum_{s=1}^j (s-1) \leq \eta_1 |\log\beta_2| \frac{j^2}{2}.
```

For case 2, we bound the LHS directly: $`\sum_{t=j+1}^{T}\eta_t L_t \leq \eta_1 s_0 \max_{s \lt s_0}|L_s|`$ and in this range $`T-j \lt j`$, so

```math
- \sum_{s=1}^{T-j} \eta_{s+j}L_s \leq \eta_1 |\log \beta_2| \sum_{s=1}^{T-j} (s-1) \leq \eta_1 |\log\beta_2|\frac{j^2}{2}. \qquad\Box
```

We now finish the proof as in Xie and Li. Expanding the recursive definitions write $`m_t = (1-\beta_1) \sum_{j=0}^t \beta_1^j g_{t-j}`$, $`v_t = (1-\beta_2) \sum_{j=0}^t \beta_2^j g^2_{t-j}`$. The weights $`(1-\beta_1)\beta_1^j`$ sum up to at most one, so by Cauchy-Schwarz

```math
m_t^2 \leq (1-\beta_1) \sum_{j=0}^t \beta^j_1g_{t-j}^2.
```

It follows that

```math
\Delta_t^2 = \frac{m_t^2}{v_t} \leq \frac{1-\beta_1}{1-\beta_2} \left(\frac{\sum_{j=0}^t\beta_1^j g_{t-j}^2}{v_t/(1-\beta_2)}\right).
```

The bracketed quantity is a ratio of exponential moving averages with memory $`\beta_1\leq \beta_2`$. If $`g^2`$ is constant, this ratio equals unity. If $`g^2`$ is shrinking or growing, then the ratio is $`\lt 1`$ or $`\gt 1`$, respectively, since the most recent values dominate. Since $`g`$ is unbounded, no pointwise bound for $`\Delta_t`$ exists. The point is that no asymptotically-vanishing sequence $`g_t`$ increases on average, so a bound on $`\overline{\Delta}_t`$ is possible. 

It is convenient to define $`m_s, v_s, g_s = 0`$ for $`s \leq 0`$. By definition $`g_s^2 = (v_s - \beta_2 v_{s-1})/(1 - \beta_2)`$, so

```math
(1-\beta_2) \sum_{j\geq 0}\beta_1^j g_{t-j}^2 = \sum_{j\geq 0} \beta_1^j (v_{t-j} - \beta_2 v_{t-j-1}) = v_t - (\beta_2 - \beta_1)\sum_{j\geq 1} \beta_1^{j-1} v_{t-j}.
```

Thus,

```math
v_t \Delta_t^2 = m_t^2 \leq (1-\beta_1)\sum_{j\geq 0} \beta_1^j g_{t-j}^2 = \frac{1-\beta_1}{1-\beta_2} \left(v_t - (\beta_2 - \beta_1)\sum_{j\geq 1}\beta_1^{j-1}v_{t-j}\right)
```

or, dividing through by $`v_t`$,
```math
\Delta_t^2 \leq \frac{1-\beta_1}{1-\beta_2}\left(1 - (\beta_2 - \beta_1)\sum_{j\geq 1}\beta_1^{j-1}\frac{v_{t-j}}{v_t}\right).
```

Using the identities $`(1-\beta_1)/(1-\beta_2) = 1 + (\beta_2 - \beta_1)/(1-\beta_2)`$ and $`(1-\beta_1) \sum_{j\geq 1} \beta_1^{j-1} = 1`$,

```math
\Delta_t^2 \leq 1 + \frac{(\beta_2 - \beta_1)(1-\beta_1)}{1-\beta_2} \sum_{j\geq 1} \beta_1^{j-1} \left(1 - \frac{v_{t-j}}{v_t}\right) = 1 + \frac{\beta_2 - \beta_1}{1-\beta_2}\left(\beta_1^{t-1} + (1-\beta_1)\sum_{j=1}^{t-1}\beta_1^{j-1}\left(1 - \frac{v_{t-j}}{v_t}\right)\right)
```

Since $`1-x \leq -\log x`$ for $`x \gt 0`$, $`1- (v_{t-j}/v_t) \leq L_t - L_{t-j}`$, and we get

```math
\Delta_t^2\leq 1 + \frac{\beta_2 - \beta_1}{1-\beta_2}\left(\beta_1^{t-1} + (1 - \beta_1)\sum_{j=1}^{t-1}\beta_1^{j-1}(L_t - L_{t-j})\right)
```

Now sum over $`t`$ with weight $`\eta_t`$:

```math
\sum_{t\leq T} \eta_t \Delta_t^2 \leq S_T + \frac{\beta_2 - \beta_1}{1-\beta_2}\left(\sum_{t\leq T}\eta_t\beta_1^{t-1} + (1-\beta_1)\sum_{t\leq T}\sum_{j=1}^{t-1}\beta_1^{j-1}\eta_t(L_t - L_{t-j})\right)
```

The double sum is over lattice points within a triangle. The same domain has an equivalent representation $`\sum_{t\leq T}\sum_{j=1}^{t-1} = \sum_{j=1}^{T-1}\sum_{t=j+1}^T`$; combined with the Lemma and the bound $`\sum_{t\leq T}\eta_t\beta_1^{t-1}\leq \eta_1/(1-\beta_1)`$,

```math
\sum_{t\leq T}\eta_t\Delta_t^2 \leq S_T + \frac{\beta_2-\beta_1}{1-\beta_2}\left(\frac{\eta_1}{1-\beta_1} + (1-\beta_1)\sum_{j=1}^{T-1}\beta_1^{j-1}C_j \right)
```

Now $`C_j = O(j^2)`$, so $`\sum_{j=1}^{T-1}\beta_1^{j-1}C_j \leq C\int_1^T\text{d}t~\beta_1^{t-1} t^2`$ and is manifestly finite for $`\beta_1 \lt 1`$. In the limit that $`T\to\infty`$, the integral can be evaluated exactly to give $`\Gamma(3, -\log \beta_1)/(\beta_1(-\log\beta_1)^3) \lt \infty`$. It follows that by Jensen's inequality,
```math
\overline{\Delta}_T^2 \leq \frac{1}{S_T}\sum_{t\leq T}\eta_t \Delta_t^2 \leq 1 + \frac{O(1)}{S_T}.
```

Since this is true for every component $`i`$ of $`\overline{\Delta}_T`$, it follows that for components $`i`$ for which $`g_{\infty, i}\to 0`$, $`\overline{\Delta}_{\infty, i}^2 = \lim_{T\to\infty} \overline{\Delta}_{T, i}^2 \leq 1`$; in particular $`|\overline{\Delta}_{\infty, i}| \leq 1`$. $`\qquad\Box`$

## 2. Implicit Bias of Adam-atan2

**Corollary.** (Implicit Bias of Adam-atan2 in Full Batch Setting) For any continuously differentiable objective function $`L: \mathbb{R}^d \to \mathbb{R}`$, $`\beta_1\leq \beta_2 \lt 1`$, initialization $`x_0`$ and non-increasing learning rate $`\{\eta_t\}_{t=1}^\infty`$ such that $`\sum_{t}\eta_t = \infty`$, if the iterates of Adam-atan2 on $`L`$ converge to some $`x_\infty`$, then $`x_\infty`$ is a KKT point of the constrained optimization problem

```math
\min_{||x||_\infty \leq 1/\lambda} L(x).
```

If $`L`$ is additionally convex, then Adam-atan2 converges to the constrained minimizer,

```math
x_\infty \in \text{argmin}_{||x||_\infty \leq 1/\lambda} L(x).
```

*Proof.* The proof relies structurally on the same arguments as Xie and Li, Theorem 1.1. We prove that the limit point of Adam-atan2 satisfies the conditions of Lemma 3.8 by showing that $`|\overline{\Delta}_{\infty, i}| \leq 1`$ for all components $`i`$, from which the Corollary follows.

We continue off at the point of analyzing the behavior of the asymptotic gradient $`g_{\infty, i}`$. Again for simplicity we fix and drop the index $`i`$. Let $`u_t = m_t/\sqrt{v_t} \to u_\infty`$ and $`\Delta_t = \frac{4}{\pi}\arctan u_t`$. Note that the AdamW theorem ([Section 1](#1-implicit-bias-of-adamw)) bounds the second moment of what we denote here as $`u_t`$. If $`g_{\infty}\neq 0`$, then 
```math
\Delta_{\infty} = \frac{4}{\pi}\arctan u_{\infty} \to \frac{4}{\pi} \arctan \text{sign}(g_{\infty}) = \text{sign}(g_{\infty})
```

and so $`|\Delta_{\infty}| = 1`$. If $`g_{\infty} = 0`$, we endeavor to bound $`\Delta_\infty`$ directly. Let $`\varphi(w) = \frac{4}{\pi}\arctan \sqrt{w}`$. It has second derivative
```math
\varphi''(w) = - \frac{1+3w}{\pi w^{3/2}(1+w)^2} \lt 0
```

for $`w \gt 0`$, so $`\varphi(w)`$ is concave in the same domain. By Jensen's inequality and the bound on the second moment of $`u_t`$ in the AdamW theorem ([Section 1](#1-implicit-bias-of-adamw)),

```math
\begin{aligned}
|\overline{\Delta}_T| = \left|\frac{1}{S_T} \sum_{t\leq T} \eta_t\Delta_t\right| &\leq \frac{1}{S_T} \sum_{t\leq T}\eta_t\varphi(u_t^2) \\
&\leq \varphi\left(\frac{1}{S_T} \sum_{t\leq T} \eta_t u_t^2\right) \leq \varphi\left(1 + \frac{O(1)}{S_T}\right) = 1 + \frac{\varphi'(1)\cdot O(1)}{S_T} + O(S_T^{-2}).
\end{aligned}
```

Taking the limit as $`T\to \infty`$ we conclude $`|\overline{\Delta}_\infty| \leq 1`$. The rest of the proof follows as in the AdamW theorem. $`\qquad\Box`$

## 3. Full-Batch GD with Nesterov Momentum

**Theorem.** Let $`L: \mathbb{R}^d \to \mathbb{R}`$ be a continuously differentiable objective function, $`\beta \in [0,1)`$ the momentum parameter, $`\lambda`$ the weight decay coefficient, $`x_0, v_0 \in \mathbb{R}^d`$ parameter initializations and $`\{\eta_t\}_{t=0}^\infty`$ (not necessarily non-increasing) learning schedule such that $`\eta_t \geq 0`$ and $`\sum_{t}\eta_t = \infty`$. Consider full-batch GD with weight decay and Nesterov (look-ahead) momentum

```math
\begin{aligned}
v_{t+1} &= \beta v_t + \nabla L(x_t) + \lambda x_t\\
x_{t+1} &= x_t - \eta_t (\nabla L(x_t) + \lambda x_t + \beta v_{t+1})
\end{aligned}
```

If $`x\to x_\infty`$ converges under this algorithm, then

1. $`\nabla L(x_\infty) + \lambda x_\infty = 0`$ and $`x_\infty`$ is a stationary point of $`L(x) + \frac{\lambda}{2}||x||_2^2`$.

2. With $`R = ||x_\infty||_2`$, $`x_\infty`$ is a KKT point of the $`\ell_2`$-norm constrained optimization problem

```math
\min_{||x||_2 \leq R} L(x).
```

3. Further, if $`L`$ is convex, $`x_\infty \in \text{argmin}_{||x||_2\leq R}L(x)`$ is a global minimizer.

Note that the $`\ell_2`$-norm $`R`$ of $`x_\infty`$ is unbounded, in contrast to the $`\ell_\infty`$ case where the norm is bounded by $`1/\lambda \lt \infty`$.

*Proof.* 

1. Define $`c_t = \nabla L(x_t) + \lambda x_t`$ and $`u_t = c_t + \beta v_{t+1}`$. Since $`\nabla L`$ is continuous, $`c_t \to c_\infty = \nabla L(x_\infty) + \lambda x_\infty`$ as $`t\to\infty`$. The recursion on momentum $`v_t`$ says $`v_{t+1} = \beta v_t + c_t`$, so

```math
v_{t+1} = \beta^{t+1}v_0 + \sum_{i=0}^{t} \beta^{t-i}c_i.
```

Using $`c_\infty/(1-\beta) = \sum_{i\geq 0}\beta^i c_\infty = \sum_{i=0}^{t}\beta^i c_\infty + \sum_{i \gt t}\beta^i c_\infty`$,

```math
v_{t+1} - \frac{c_\infty}{1-\beta} = \beta^{t+1}v_0 + \sum_{i=0}^{t}\beta^{t-i}(c_i - c_\infty) - \frac{\beta^{t+1}}{1-\beta}c_\infty.
```

The first and last term vanish as $`t\to\infty`$ since $`\beta \lt 1`$. For the middle term, let $`M = \sup_i ||c_i - c_\infty|| \lt \infty`$, finite because $`c_i \to c_\infty`$. For every $`\epsilon \gt 0`$ let $`N`$ be such that $`||c_i - c_\infty|| \leq \epsilon`$ for all $`i \geq N`$. Then for $`t \geq N`$:

```math
\left|\left|\sum_{i=0}^{t}\beta^{t-i}(c_i - c_\infty)\right|\right| \leq M\sum_{i \lt N}\beta^{t-i} + \epsilon\sum_{i=N}^{t}\beta^{t-i} \leq \frac{M\beta^{t-N+1}}{1-\beta} + \frac{\epsilon}{1-\beta},
```

and the first term vanishes for sufficiently-large $`t`$. Therefore, $`v_t \to c_\infty/(1-\beta)`$. It follows that

```math
u_t = c_t + \beta v_{t+1} \to c_\infty + \frac{\beta c_\infty}{1-\beta} = \frac{c_\infty}{1-\beta}.
```

Suppose for the sake of contradiction that $`c_\infty = \nabla L(x_\infty) + \lambda x_\infty \neq 0`$. First,

```math
\langle c_\infty, u_t\rangle \to \frac{||c_\infty||_2^2}{1-\beta} \gt 0.
```

Thus, there exists $`s`$ such that $`\langle c_\infty, u_t\rangle \geq ||c_\infty||_2^2/(2(1-\beta))`$ for all $`t \geq s`$. Then

```math
\langle c_\infty, x_0 - x_T\rangle = \sum_{t=0}^{T-1} \eta_t \langle c_\infty, u_t\rangle \geq \sum_{t=0}^{s-1}\eta_t \langle c_\infty, u_t\rangle + \frac{||c_\infty||_2^2}{2(1-\beta)} \sum_{i=s}^{T-1} \eta_i.
```

This quantity goes to $`\infty`$ as $`T\to \infty`$, contradicting the assumption that $`x_t`$ converges to (finite) $`x_\infty`$. Therefore $`c_\infty = 0`$.

2. Let $`R = ||x_\infty||_2`$ and

```math
g(x) = \frac{1}{2}\left( ||x||_2^2 - R^2 \right).
```

The Lagrangian for the constrained optimization problem is given by

```math
\mathcal{L}(x, \mu) = L(x) + \mu g(x).
```

Primal feasibility $`g(x_\infty) \leq 0`$ is true by definition of $`R`$. Stationarity $`\nabla L(x_\infty) + \mu x_\infty = 0`$ is true for dual variable $`\mu^{\ast} = \lambda \gt 0`$, which also satisfies dual feasibility. Complementary slackness $`\mu g(x_\infty) = 0`$ holds since $`g(x_\infty) = 0`$.

3. $`L`$ and the constraint are convex, so KKT conditions are sufficient for a global minimum.$`\qquad \Box`$
