# RungeKuttaToolKit.jl

**Copyright © 2019-2024 by David K. Zhang. Released under the [MIT License][1].**

**RungeKuttaToolKit.jl** is a Julia package that provides tools for constructing and analyzing Runge–Kutta methods. It includes optimized, SIMD-accelerated code for enumerating rooted trees, building instruction tables, and computing Butcher weights $\Phi_t(A)$ and their derivatives.

> The algorithms that underlie **RungeKuttaToolKit.jl** are described in **[this paper][5]**, which is **[freely available in read-only form][6]**.

**RungeKuttaToolKit.jl** is closely related to its sister repository, **[RKTK][2]**.

* **RungeKuttaToolKit.jl** (this repository) implements the core mathematical operations of the algebraic theory of Runge–Kutta methods, such as computing Butcher weights $\Phi_t(A)$ and evaluating the gradient of the sum-of-squares residual function. **RungeKuttaToolKit.jl** is intended to be useful to all researchers studying Runge–Kutta methods and is [registered in the Julia General package registry][4], so it can be installed into any Julia environment with the command `]add RungeKuttaToolKit`.

* **[RKTK][2]** contains parallel search and nonlinear optimization scripts that specifically target the discovery of new Runge–Kutta methods. In contrast to the pure mathematical operations in **RungeKuttaToolKit.jl**, the programs in **RKTK** establish a number of RKTK-specific conventions regarding optimization hyperparameters and file structure. Consequently, **RKTK** is not a registered Julia package.

Both **RungeKuttaToolKit.jl** and **RKTK** are designed with reproducibility as an explicit goal. The [RKTKSearch.jl][3] program has been tested to produce bit-for-bit identical results across multiple architectures (i686, x86-64, ARMv8) and multiple CPU generations (Haswell, Skylake, Rocket Lake, Zen 5).



[1]: https://github.com/dzhang314/RungeKuttaToolKit.jl/blob/master/LICENSE
[2]: https://github.com/dzhang314/RKTK
[3]: https://github.com/dzhang314/RKTK/blob/master/RKTKSearch.jl
[4]: https://juliahub.com/ui/Packages/General/RungeKuttaToolKit
[5]: https://link.springer.com/article/10.1007/s11075-024-01783-2
[6]: https://rdcu.be/dz7sy
