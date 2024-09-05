# RungeKuttaToolKit.jl

**RungeKuttaToolKit.jl** is a Julia package for designing and analyzing [Runge--Kutta methods](https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta_methods) for solving differential equations.

--------------------------------------------------------------------------------

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

--------------------------------------------------------------------------------

## Background

To design a Runge--Kutta method, one must find a square matrix ``A \in \mathbb{R}^{s \times s}`` and a vector ``\mathbf{b} \in \mathbb{R}^s`` that satisfy the equation

```math
\mathbf{b} \cdot \Phi_t(A) = \frac{1}{\gamma(t)}

```
for as many rooted trees $t$ as possible. Here, ``\Phi_t(A)`` denotes the Butcher weight vector associated to the

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


## Usage



```@docs
RKOCEvaluator
```
