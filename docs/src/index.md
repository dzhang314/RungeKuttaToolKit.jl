# RungeKuttaToolKit.jl

**RungeKuttaToolKit.jl** is a Julia package for designing and analyzing [Runge--Kutta methods](https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta_methods) for solving [ordinary differential equations](https://en.wikipedia.org/wiki/Ordinary_differential_equation). The main tool it provides is the `RKOCEvaluator` type, which implements fast algorithms for evaluating the residuals and gradients of the Runge--Kutta order conditions

```math
\mathbf{b} \cdot \Phi_t(A) = \frac{1}{\gamma(t)}
```

over a given set of rooted trees ``t \in T``.



## Features

- **RungeKuttaToolKit.jl** is designed for **extremely high performance** and **stable memory usage**. It evaluates Runge--Kutta order conditions and their derivatives several million times per second on a single CPU core, using fast in-place algorithms that allocate zero memory outside of `RKOCEvaluator` construction.
- **RungeKuttaToolKit.jl** works with all numeric types that support arithmetic (`+` `-` `*` `/`), including low-precision and high-precision floating-point types, `BigFloat`, `Rational`, and even symbolic types.
- **RungeKuttaToolKit.jl** is lightweight and has **zero dependencies** outside of Julia `Base`.
- **RungeKuttaToolKit.jl** is [regularly and extensively tested](https://github.com/dzhang314/RungeKuttaToolKit.jl/blob/master/test/runtests.jl) for both correctness and performance.



## Quick Start

To install **RungeKuttaToolKit.jl**, run `]add RungeKuttaToolKit` in the Julia REPL.

```julia
using RungeKuttaToolKit

order = 4
num_stages = 4

# RK4 Butcher tableau
A = [0.0 0.0 0.0 0.0;
     0.5 0.0 0.0 0.0;
     0.0 0.5 0.0 0.0;
     0.0 0.0 1.0 0.0]
b = [1/6, 1/3, 1/3, 1/6]

ev = RKOCEvaluator(order, num_stages)
residuals = ev(A, b)
println(residuals) # This should be close to zero,
# which confirms that RK4 is a fourth-order method.

# An RKOCEvaluator can also be constructed from a set of
# rooted trees. Here, we select only the trees of order 5.
trees = rooted_trees(order + 1)
sigma = inv.(Float64.(butcher_symmetry.(trees)))
err_ev = RKOCEvaluator(trees, num_stages)
error_coeffs = sigma .* err_ev(A, b)
println(error_coeffs) # This computes the principal error
# coefficients of RK4, which should all be nonzero.
```



### Elementary Differentials

If you've studied calculus, you probably know that any analytic function $f: \R \to \R$ can be written as a _Taylor series_. For all $x \in \R$ and sufficiently small $h \in \R$, we have:

```math
f(x + h) = f(x) + hf'(x) + \frac{h^2}{2}f''(x) + \frac{h^3}{6}f'''(x) + \cdots + \frac{h^n}{n!}f^{(n)}(x) + \cdots
```

In the study of differential equations, we use a special form of Taylor series called a _Butcher series_, named after the mathematician [John C. Butcher](https://en.wikipedia.org/wiki/John_C._Butcher). Unlike a Taylor series, whose terms are indexed by a natural number $n \in \N$, the terms of a Butcher series are indexed by _rooted trees_.

To understand the structure of a Butcher series, let's say we're trying to analyze a system of differential equations.

```math
\begin{aligned}
y_1'(t) &= f_1(y_1(t), y_2(t), \ldots, y_n(t)) \\
y_2'(t) &= f_2(y_1(t), y_2(t), \ldots, y_n(t)) \\
&\hspace{2mm} \vdots \\
y_n'(t) &= f_n(y_1(t), y_2(t), \ldots, y_n(t))
\end{aligned}
```

Here, $y_1, y_2, \ldots, y_n: \R \to \R$ are the unknown functions we're trying to find, and $f_1, f_2, \dots, f_n: \R^n \to \R$ are the known right-hand side functions. To simplify our notation, we can rewrite this system in vector form:

```math
\vy'(t) = \vf(\vy(t))
```

Here, we have gathered the unknown functions ``y_1, y_2, \ldots, y_n`` and right-hand side functions ``f_1, f_2, \dots, f_n`` into vector-valued functions ``\vy: \R \to \R^n`` and ``\vf: \R^n \to \R^n``.

```math
\begin{aligned}
\vy' &= \vf \\
\vy'' &= \vf'[\vf] \\
\vy''' &= \vf''[\vf, \vf] + \vf'[\vf'[\vf]] \\
\vy'''' &= \vf'''[\vf, \vf, \vf] + \vf''[\vf'[\vf], \vf] + \vf''[\vf, \vf'[\vf]] + \\
&\hspace{5mm} \vf''[\vf'[\vf], \vf] + \vf'[\vf''[\vf, \vf]] + \vf'[\vf'[\vf'[\vf]]]
\end{aligned}
```

**Definition:** Let ``t`` be a rooted tree. The **_extension_** of ``t``, denoted by ``[t]``, is the rooted tree obtained from ``t`` by adjoining a new vertex ``v``, adding a new edge from ``v`` to the root of ``t``, and designating ``v`` as the root of ``[t]``.

**Definition:** Let ``s`` and ``t`` be rooted trees. The **_rooted sum_** of ``s`` and ``t``, denoted by ``s \hash t``, is the rooted tree obtained by taking the disjoint union of ``s`` and ``t`` and merging their roots into a single vertex, which is designated as the root of ``s \hash t``.

```@raw html
<p><strong>Example:</strong> Suppose <span>$s =$</span>
<svg xmlns="http://www.w3.org/2000/svg" width="41" height="91" viewBox="-20.5 -83 41 91" style="vertical-align: middle;">
<line x1="0" y1="0" x2="0" y2="-25" stroke="white" stroke-width="5.0" />
<line x1="0" y1="0" x2="0" y2="-25" stroke="black" stroke-width="3.0" />
<line x1="0" y1="-25" x2="0" y2="-50" stroke="white" stroke-width="5.0" />
<line x1="0" y1="-25" x2="0" y2="-50" stroke="black" stroke-width="3.0" />
<line x1="0" y1="-50" x2="-12.5" y2="-75" stroke="white" stroke-width="5.0" />
<line x1="0" y1="-50" x2="-12.5" y2="-75" stroke="black" stroke-width="3.0" />
<line x1="0" y1="-50" x2="12.5" y2="-75" stroke="white" stroke-width="5.0" />
<line x1="0" y1="-50" x2="12.5" y2="-75" stroke="black" stroke-width="3.0" />
<circle cx="0" cy="0" r="7" stroke="white" />
<circle cx="0" cy="-25" r="7" stroke="white" />
<circle cx="0" cy="-50" r="7" stroke="white" />
<circle cx="-12.5" cy="-75" r="7" stroke="white" />
<circle cx="12.5" cy="-75" r="7" stroke="white" />
</svg>
and <span>$t =$</span>
<svg xmlns="http://www.w3.org/2000/svg" width="41" height="66" viewBox="-20.5 -58 41 66" style="vertical-align: middle;">
<line x1="0" y1="0" x2="-12.5" y2="-25" stroke="white" stroke-width="5.0" />
<line x1="0" y1="0" x2="-12.5" y2="-25" stroke="black" stroke-width="3.0" />
<line x1="-12.5" y1="-25" x2="-12.5" y2="-50" stroke="white" stroke-width="5.0" />
<line x1="-12.5" y1="-25" x2="-12.5" y2="-50" stroke="black" stroke-width="3.0" />
<line x1="0" y1="0" x2="12.5" y2="-25" stroke="white" stroke-width="5.0" />
<line x1="0" y1="0" x2="12.5" y2="-25" stroke="black" stroke-width="3.0" />
<circle cx="0" cy="0" r="7" stroke="white" />
<circle cx="-12.5" cy="-25" r="7" stroke="white" />
<circle cx="-12.5" cy="-50" r="7" stroke="white" />
<circle cx="12.5" cy="-25" r="7" stroke="white" />
</svg>.
Then <span>$[t] =$</span>
<svg xmlns="http://www.w3.org/2000/svg" width="41" height="91" viewBox="-20.5 -83 41 91" style="vertical-align: middle;">
<line x1="0" y1="0" x2="0" y2="-25" stroke="white" stroke-width="5.0" />
<line x1="0" y1="0" x2="0" y2="-25" stroke="black" stroke-width="3.0" />
<line x1="0" y1="-25" x2="-12.5" y2="-50" stroke="white" stroke-width="5.0" />
<line x1="0" y1="-25" x2="-12.5" y2="-50" stroke="black" stroke-width="3.0" />
<line x1="-12.5" y1="-50" x2="-12.5" y2="-75" stroke="white" stroke-width="5.0" />
<line x1="-12.5" y1="-50" x2="-12.5" y2="-75" stroke="black" stroke-width="3.0" />
<line x1="0" y1="-25" x2="12.5" y2="-50" stroke="white" stroke-width="5.0" />
<line x1="0" y1="-25" x2="12.5" y2="-50" stroke="black" stroke-width="3.0" />
<circle cx="0" cy="0" r="7" stroke="white" />
<circle cx="0" cy="-25" r="7" stroke="white" />
<circle cx="-12.5" cy="-50" r="7" stroke="white" />
<circle cx="-12.5" cy="-75" r="7" stroke="white" />
<circle cx="12.5" cy="-50" r="7" stroke="white" />
</svg>
and <span>$s \hash t =$</span>
<svg xmlns="http://www.w3.org/2000/svg" width="91" height="91" viewBox="-45.5 -83 91 91" style="vertical-align: middle;">
<line x1="0" y1="0" x2="-25" y2="-25" stroke="white" stroke-width="5.0" />
<line x1="0" y1="0" x2="-25" y2="-25" stroke="black" stroke-width="3.0" />
<line x1="-25" y1="-25" x2="-25" y2="-50" stroke="white" stroke-width="5.0" />
<line x1="-25" y1="-25" x2="-25" y2="-50" stroke="black" stroke-width="3.0" />
<line x1="-25" y1="-50" x2="-37.5" y2="-75" stroke="white" stroke-width="5.0" />
<line x1="-25" y1="-50" x2="-37.5" y2="-75" stroke="black" stroke-width="3.0" />
<line x1="-25" y1="-50" x2="-12.5" y2="-75" stroke="white" stroke-width="5.0" />
<line x1="-25" y1="-50" x2="-12.5" y2="-75" stroke="black" stroke-width="3.0" />
<line x1="0" y1="0" x2="12.5" y2="-25" stroke="white" stroke-width="5.0" />
<line x1="0" y1="0" x2="12.5" y2="-25" stroke="black" stroke-width="3.0" />
<line x1="12.5" y1="-25" x2="12.5" y2="-50" stroke="white" stroke-width="5.0" />
<line x1="12.5" y1="-25" x2="12.5" y2="-50" stroke="black" stroke-width="3.0" />
<line x1="0" y1="0" x2="37.5" y2="-25" stroke="white" stroke-width="5.0" />
<line x1="0" y1="0" x2="37.5" y2="-25" stroke="black" stroke-width="3.0" />
<circle cx="0" cy="0" r="7" stroke="white" />
<circle cx="-25" cy="-25" r="7" stroke="white" />
<circle cx="-25" cy="-50" r="7" stroke="white" />
<circle cx="-37.5" cy="-75" r="7" stroke="white" />
<circle cx="-12.5" cy="-75" r="7" stroke="white" />
<circle cx="12.5" cy="-25" r="7" stroke="white" />
<circle cx="12.5" cy="-50" r="7" stroke="white" />
<circle cx="37.5" cy="-25" r="7" stroke="white" />
</svg>.
</p>
```



## Usage

```@docs
RKOCEvaluator
```
