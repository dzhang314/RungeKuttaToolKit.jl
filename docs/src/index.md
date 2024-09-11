# RungeKuttaToolKit.jl

**RungeKuttaToolKit.jl** is a Julia package for designing and analyzing [Runge--Kutta methods](https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta_methods) for solving [ordinary differential equations](https://en.wikipedia.org/wiki/Ordinary_differential_equation). The main tool it provides is the `RKOCEvaluator` type, which implements fast algorithms for evaluating the residuals and gradients of the Runge--Kutta order conditions

```math
\mathbf{b} \cdot \Phi_t(A) = \frac{1}{\gamma(t)}
```

over a given set of rooted trees ``t \in T``.



## Features

- **RungeKuttaToolKit.jl** is designed for **extremely high performance** and **stable memory usage**. It evaluates Runge--Kutta order conditions and their derivatives millions of times per second on a single CPU core, using fast in-place algorithms that allocate zero memory outside of `RKOCEvaluator` construction.
- **RungeKuttaToolKit.jl** works with **all numeric types** that support arithmetic (`+` `-` `*` `/`), including low-precision and high-precision floating-point types, `BigFloat`, `Rational`, and even symbolic types.
- **RungeKuttaToolKit.jl** is lightweight and has **zero dependencies** outside of Julia `Base`.
- **RungeKuttaToolKit.jl** is [**regularly and extensively tested**](https://github.com/dzhang314/RungeKuttaToolKit.jl/blob/master/test/runtests.jl) for correctness and performance.



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



## Generating Rooted Trees

The first step in using **RungeKuttaToolKit.jl** is generating a list of rooted trees. Each Runge--Kutta order condition ``\mathbf{b} \cdot \Phi_t(A) = 1/\gamma(t)`` corresponds to a rooted tree ``t``. For a Runge--Kutta method ``(A, \mathbf{b})`` to have order ``p``, it needs to satisfy this equation for all rooted trees ``t`` having up to ``p`` vertices.

**RungeKuttaToolKit.jl** provides two functions for generating rooted trees: `rooted_trees(n)`, which generates all rooted trees with **exactly** ``n`` vertices, and `all_rooted_trees(n)`, which generates all rooted trees with **at most** ``n`` vertices.

```@docs
rooted_trees
```

```@docs
all_rooted_trees
```



## `RKOCEvaluator` Construction

With a list of rooted trees in hand, we can construct an `RKOCEvaluator` object, which is used to evaluate the residuals and gradients of the Runge--Kutta order conditions with respect to the entries of ``A`` and ``\mathbf{b}``. The `RKOCEvaluator` constructor analyzes the given list of rooted trees to identify common subtrees whose Butcher weight vectors ``\Phi_t(A)`` can be reused to eliminate redundant computation.

```@docs
RKOCEvaluator
```



## `RKOCEvaluator` Usage

An `RKOCEvaluator` object can be used to perform seven operations which are summarized below. A full description of each operation follows the summary.

- `ev([residuals,] A, b)`: Compute residuals.
- `ev(cost, A, b)`: Compute cost function.
- `ev'([dresiduals,] A, dA, b, db)`: Compute directional derivatives.
     of residuals
- `ev'([dresiduals,] A, i, j, b)`: Compute partial derivatives of residuals.
     with respect to ``A_{i,j}``
- `ev'([dresiduals,] A, i)`: Compute partial derivatives of residuals.
     with respect to ``b_i``
- `ev'([dA, db,] cost, A, b)`: Compute gradient of cost function.
- `ev'([jacobian, A, b,] param, x)`: Compute Jacobian of residuals
     with respect to parameterization.

Every operation that returns a matrix or vector has an **allocating version** that returns its result in a new array and an **in-place version** that writes its result to an existing array. We use square brackets to denote optional output arguments. For example, the operation `ev([residuals,] A, b)` can be called as `ev(A, b)` (allocating) or `ev(residuals, A, b)` (in-place).

As a mnemonic device, the operations that compute derivatives are called with a single quote `'` following the `RKOCEvaluator` object, emulating the prime symbol ``f'(x)`` used to denote derivatives in mathematical notation. This distinguishes operations like `ev(cost, A, b)` and `ev'(cost, A, b)` that would otherwise have identical function signatures.

```@docs
RungeKuttaToolKit.AbstractRKOCEvaluator
```

```@docs
RungeKuttaToolKit.AbstractRKOCAdjoint
```



## Cost Functions

```@docs
RKCostL1
```

```@docs
RKCostWeightedL1
```

```@docs
RKCostL2
```

```@docs
RKCostWeightedL2
```

```@docs
RKCostLInfinity
```

Derivatives are not yet implemented for `RKCostLInfinity`.

```@docs
RKCostWeightedLInfinity
```

Derivatives are not yet implemented for `RKCostWeightedLInfinity`.

```@docs
RKCostHuber
```

```@docs
RKCostWeightedHuber
```



## Parameterizations

```@docs
RKParameterizationExplicit
```

```@docs
RKParameterizationDiagonallyImplicit
```

```@docs
RKParameterizationImplicit
```

```@docs
RKParameterizationParallelExplicit
```
