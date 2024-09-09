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



## Usage

```julia
ev = RKOCEvaluator(order::Integer, num_stages::Integer) # T = Float64
ev = RKOCEvaluator{T}(order::Integer, num_stages::Integer)
```

The type `T` needs to support `+`, `-`, `*`, and `inv`

```julia
trees = rooted_trees(n::Integer)
trees = rooted_trees(n::Integer; tree_ordering::Symbol)
```

**RungeKuttaToolKit.jl** supports several methods for generating rooted trees

- `:reverse_lexicographic` is the default and most efficient ordering, which uses a [fast algorithm by Terry Beyer and Sandra Mitchell Hedetniemi](https://epubs.siam.org/doi/pdf/10.1137/0209055) to generate rooted trees in constant time per tree.
- `:lexicographic`:
- `:attach` generates rooted trees of order ``n`` by attaching a leaf in all possible positions to all rooted trees of order ``n-1``, starting by attaching a leaf to the root and proceeding depth-first. This is the ordering usually produced by paper-and-pencil application of the product rule to the set of elementary differentials.
- `:butcher` uses [Algorithm 3](https://link.springer.com/content/pdf/10.1007/978-3-030-70956-3_2) from [John Butcher's monograph on B-Series](https://link.springer.com/content/pdf/10.1007/978-3-030-70956-3.pdf) to generate rooted trees in lexicographic order. This is the ordering that would be produced by manually applying the product rule to the set of elementary differentials, but it is slower than `:attach`.


```julia
ev = RKOCEvaluator(trees::AbstractVector{LevelSequence}, num_stages::Integer) # T = Float64
ev = RKOCEvaluator{T}(trees::AbstractVector{LevelSequence}, num_stages::Integer)
```



```julia
residuals = ev(A, b)
ev(residuals, A, b)
```



```@docs
RKOCEvaluator
```
