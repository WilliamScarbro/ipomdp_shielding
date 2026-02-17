# Mini-Course: Convex Algebras, Probability Monads, and Linear Programming

*For researchers in formal methods, control, and probabilistic verification*

---

## Course Overview

This mini-course develops a unified mathematical understanding of:

* Convex combinations and belief envelopes
* Probability monads and categorical semantics
* Why belief propagation becomes LP / LFP
* Why convexity is preserved *before* normalization
* Why normalization breaks convexity

**Goal:** Understand why *convex algebras are exactly probability algebras*, and why *linear programming is the natural computational tool* for belief envelope propagation in interval HMMs and IPOMDPs.

Each section is designed to be approximately **2 pages of reading**, with **exercises** to ensure deep understanding.

---

# Part I — Convexity as Algebraic Structure

## 1. Convex Combinations and Convex Sets

### 1.1 Convex combinations

Let $X$ be a set. A **formal convex combination** is an expression

Abc $ \sum_{x=1} $
$ \sum_{i=1}^n p_i x_i $

where $x_i \in X$, $p_i \ge 0$, and $\sum_i p_i = 1$.

This expression represents *mixing*, *averaging*, or *probabilistic choice*.

---

### 1.2 Convex subsets

A subset $C \subseteq \mathbb{R}^n$ is **convex** if

[
x,y \in C \implies \lambda x + (1-\lambda) y \in C \quad \forall \lambda \in [0,1].
]

Convexity ensures closure under mixing.

Examples:

* Probability simplex $\Delta(S)$
* Polytopes
* Affine subspaces

---

### 1.3 Abstract convex spaces

Instead of embedding into $\mathbb{R}^n$, we define convexity *algebraically*.

A **convex algebra** is a set $X$ equipped with an operation

[
\alpha : D(X) \to X
]

that interprets formal probability distributions as elements of $X$.

Here $D(X)$ denotes the set of **finite probability distributions** on $X$.

---

### 1.4 Convex algebra axioms

For
[
\alpha\left(\sum_i p_i \delta_{x_i}\right) = \sum_i p_i x_i
]

the following axioms must hold:

* **Unit:** $1\cdot x = x$
* **Associativity:**

[
\sum_i p_i \left( \sum_j q_{ij} x_{ij} \right)
= \sum_{i,j} p_i q_{ij} x_{ij}
]

This expresses that *probabilistic mixing is associative*.

---

### Exercises

1. Show that $\mathbb{R}^n$ is a convex algebra.
2. Show that the probability simplex $\Delta(S)$ is a convex algebra.
3. Prove that any convex subset of $\mathbb{R}^n$ is a convex algebra.
4. Give an example of a set that cannot be made into a convex algebra.

---

# Part II — The Finite Distribution Monad

## 2. Probability Distributions as a Monad

### 2.1 The distribution functor

Define a functor

[
D : \mathbf{Set} \to \mathbf{Set}
]

by

[
D(X) = { \text{finite probability distributions over } X }.
]

Elements are formal convex combinations:

[
\sum_i p_i \delta_{x_i}.
]

---

### 2.2 Monad structure

The monad consists of:

* **Unit:** $\eta_X(x) = \delta_x$
* **Multiplication:** $\mu_X : D(D(X)) \to D(X)$

[
\sum_i p_i \left( \sum_j q_{ij} \delta_{x_{ij}} \right)
\mapsto
\sum_{i,j} p_i q_{ij} \delta_{x_{ij}}.
]

This is **law of total probability**.

---

### 2.3 Probabilistic computation

The Kleisli category of $D$ has:

* Objects: sets
* Morphisms: probabilistic kernels $f : X \to D(Y)$

Composition is Bayesian marginalization.

This provides a **categorical semantics of probabilistic computation**.

---

### Exercises

1. Prove that $(D,\eta,\mu)$ satisfies the monad laws.
2. Show that deterministic functions embed as probabilistic kernels.
3. Show that stochastic matrices define Kleisli morphisms.
4. Write out Kleisli composition explicitly for HMM transitions.

---

# Part III — Convex Algebras as Monad Algebras

## 3. Eilenberg–Moore Algebras

An **algebra for monad $D$** is a function

[
\alpha : D(X) \to X
]

satisfying:

* $\alpha(\delta_x) = x$
* $\alpha( D(\alpha)(\nu) ) = \alpha( \mu(\nu) )$

---

### 3.1 Interpretation

$\alpha$ interprets probability distributions as *their barycenter*.

This is exactly what convex combinations do.

---

### 3.2 The equivalence theorem

> **Theorem:** Convex algebras are precisely Eilenberg–Moore algebras of $D$.

This establishes:

[
\text{Convex spaces} \cong D\text{-algebras}.
]

---

### 3.3 Examples

* $\mathbb{R}^n$
* Belief simplices
* Convex polytopes
* Credal sets

---

### Exercises

1. Verify the algebra laws for $\mathbb{R}^n$.
2. Show that belief propagation is Kleisli composition.
3. Show normalization is *not* a $D$-algebra morphism.

---

# Part IV — Belief Propagation Geometry

## 4. Linear Propagation and Projective Collapse

### 4.1 Unnormalized update

[
u = O_z T^T b]

This is linear in $T$ and $b$.

---

### 4.2 Normalization

[
b^+ = \frac{\nu}{\mathbf{1}^T \nu}
]

This is a **projective transformation**.

---

### 4.3 Convexity preservation

* Unnormalized belief envelopes are convex.
* Normalized belief envelopes are generally nonconvex.

---

### 4.4 Lifted homogeneous coordinates

Define $t = \mathbf{1}^T \nu$ and lift:

[
(\nu, t) \in \mathbb{R}^{n+1}
]

Beliefs are recovered by projection:

[
b = \frac{\nu}{t}
]

---

### Exercises

1. Construct a counterexample showing normalized envelopes are nonconvex.
2. Show lifted envelopes are convex.
3. Interpret normalization as projective geometry.

---

# Part V — Why Linear Programming Works

## 5. Linear-Fractional Optimization

We wish to compute:

[
\max_{T \in \mathcal{T}} c^T b^+
= \max_{\nu \in \mathcal{U}} \frac{c^T \nu}{\mathbf{1}^T \nu}
]

This is a **linear-fractional program (LFP)**.

---

### 5.1 Charnes–Cooper transformation

Let:

[
y = \frac{\nu}{t}, \quad \tau = \frac{1}{t}
]

Then $\nu = y / \tau$, and constraints become linear.

---

### 5.2 Envelope propagation via LP

This yields LPs computing:

* belief component bounds
* state subset occupancy lower bounds
* admissible actions

---

### 5.3 Why LP is optimal

Because:

* uncertainty sets are convex
* belief update is linear-fractional
* extrema occur at polytope vertices

---

### Exercises

1. Derive the Charnes–Cooper transformation.
2. Implement belief envelope LP for a 2-state HMM.
3. Prove that envelope extrema occur at vertices.

---

# Part VI — Conceptual Synthesis

## 6. Why This Framework Is Inevitable

Convex algebras are the **only structures compatible with probability**.

LP arises naturally because:

* probability is linear
* uncertainty sets are convex
* extrema are linear programs

Your shielding work operates at the intersection of:

[
\text{Category theory} \cap \text{Convex geometry} \cap \text{Optimization}
]

---

### Final Exercises

1. Show belief envelopes form a convex algebra before normalization.
2. Prove normalization breaks algebra morphisms.
3. Formalize your belief polytope method as a convex algebra computation.

---

# Further Reading

* Tobias Fritz — *Convex Spaces I: Algebraic Foundations*
* Fabio Cozman — *Credal Networks*
* Jacobs — *Convexity, Duality and Effects*
* Boyd & Vandenberghe — *Convex Optimization*

---

**End of Mini-Course**
