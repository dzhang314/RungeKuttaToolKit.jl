using MultiFloats
using RungeKuttaToolKit
using StatsBase: sample
using Test


const MAX_ORDER = 10
const MAX_NUM_STAGES = 20
const NUM_RANDOM_TRIALS = 10
const NUMERIC_TYPES = [
    Float16, Float32, Float64, BigFloat,
    Float64x1, Float64x2, Float64x3, Float64x4,
    Float64x5, Float64x6, Float64x7, Float64x8]


const ALL_TREES = all_rooted_trees(MAX_ORDER)
@inline random_trees() = sample(
    ALL_TREES, rand(1:length(ALL_TREES));
    replace=false, ordered=false)


function relative_difference(x::T, y::T) where {T}
    if iszero(x) & iszero(y)
        return zero(T)
    else
        half_diff = abs(x - y) / (abs(x) + abs(y))
        return half_diff + half_diff
    end
end


function maximum_relative_difference(
    xs::AbstractArray{T},
    ys::AbstractArray{T},
) where {T}
    @assert axes(xs) == axes(ys)
    result = zero(T)
    for (x, y) in zip(xs, ys)
        diff = relative_difference(x, y)
        if diff > result
            result = diff
        end
    end
    return result
end


############################################################## RESHAPE OPERATORS


using RungeKuttaToolKit: reshape_explicit!,
    reshape_diagonally_implicit!, reshape_implicit!


@testset "reshape operators" begin
    let
        A, b = reshape_explicit!(
            Matrix{BigFloat}(undef, 3, 3),
            Vector{BigFloat}(undef, 3),
            BigFloat[1, 2, 3, 4, 5, 6])
        @test A == BigFloat[0 0 0; 1 0 0; 2 3 0]
        @test b == BigFloat[4, 5, 6]
    end
    let
        x = reshape_explicit!(
            Vector{BigFloat}(undef, 6),
            BigFloat[0 0 0; 1 0 0; 2 3 0],
            BigFloat[4, 5, 6])
        @test x == BigFloat[1, 2, 3, 4, 5, 6]
    end
    let
        A = reshape_explicit!(
            Matrix{BigFloat}(undef, 3, 3),
            BigFloat[1, 2, 3])
        @test A == BigFloat[0 0 0; 1 0 0; 2 3 0]
    end
    let
        x = reshape_explicit!(
            Vector{BigFloat}(undef, 3),
            BigFloat[0 0 0; 1 0 0; 2 3 0])
        @test x == BigFloat[1, 2, 3]
    end
    let
        A, b = reshape_diagonally_implicit!(
            Matrix{BigFloat}(undef, 3, 3),
            Vector{BigFloat}(undef, 3),
            BigFloat[1, 2, 3, 4, 5, 6, 7, 8, 9])
        @test A == BigFloat[1 0 0; 2 3 0; 4 5 6]
        @test b == BigFloat[7, 8, 9]
    end
    let
        x = reshape_diagonally_implicit!(
            Vector{BigFloat}(undef, 9),
            BigFloat[1 0 0; 2 3 0; 4 5 6],
            BigFloat[7, 8, 9])
        @test x == BigFloat[1, 2, 3, 4, 5, 6, 7, 8, 9]
    end
    let
        A = reshape_diagonally_implicit!(
            Matrix{BigFloat}(undef, 3, 3),
            BigFloat[1, 2, 3, 4, 5, 6])
        @test A == BigFloat[1 0 0; 2 3 0; 4 5 6]
    end
    let
        x = reshape_diagonally_implicit!(
            Vector{BigFloat}(undef, 6),
            BigFloat[1 0 0; 2 3 0; 4 5 6])
        @test x == BigFloat[1, 2, 3, 4, 5, 6]
    end
    let
        A, b = reshape_implicit!(
            Matrix{BigFloat}(undef, 3, 3),
            Vector{BigFloat}(undef, 3),
            BigFloat[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
        @test A == BigFloat[1 2 3; 4 5 6; 7 8 9]
        @test b == BigFloat[10, 11, 12]
    end
    let
        x = reshape_implicit!(
            Vector{BigFloat}(undef, 12),
            BigFloat[1 2 3; 4 5 6; 7 8 9],
            BigFloat[10, 11, 12])
        @test x == BigFloat[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    end
    let
        A = reshape_implicit!(
            Matrix{BigFloat}(undef, 3, 3),
            BigFloat[1, 2, 3, 4, 5, 6, 7, 8, 9])
        @test A == BigFloat[1 2 3; 4 5 6; 7 8 9]
    end
    let
        x = reshape_implicit!(
            Vector{BigFloat}(undef, 9),
            BigFloat[1 2 3; 4 5 6; 7 8 9])
        @test x == BigFloat[1, 2, 3, 4, 5, 6, 7, 8, 9]
    end
end


##################################################### RKOCEVALUATOR CONSTRUCTION


using RungeKuttaToolKit.ButcherInstructions: execute_instructions


function test_complete(::Type{T}, order::Int) where {T}
    ev = RKOCEvaluator{T}(order, 0)
    @test all(i == j for (i, j) in pairs(ev.table.selected_indices))
    @test execute_instructions(ev.table.instructions) == all_rooted_trees(order)
    @test ev.inv_gamma == inv.(T.(butcher_density.(all_rooted_trees(order))))
end


function test_incomplete(::Type{T}, order::Int) where {T}
    trees = random_trees()
    ev = RKOCEvaluator{T}(trees, 0)
    computed_trees = execute_instructions(ev.table.instructions)
    @test computed_trees[ev.table.selected_indices] == trees
    @test ev.inv_gamma == inv.(T.(butcher_density.(trees)))
end


@testset "RKOCEvaluator construction" begin
    for order = 0:MAX_ORDER
        for T in NUMERIC_TYPES
            test_complete(T, order)
            test_incomplete(T, order)
        end
    end
end


################################################### RESIDUALS AND COST FUNCTIONS


function test_residuals(::Type{T}, method::Function, order::Int) where {T}

    A, b = method(T)
    num_stages = length(b)
    @assert size(A) == (num_stages, num_stages)
    @assert size(b) == (num_stages,)
    ev = RKOCEvaluator{T}(order, num_stages)

    residuals = similar(ev.inv_gamma)
    if isbitstype(T)
        @test iszero(@allocated ev(residuals, A, b))
    else
        ev(residuals, A, b)
    end

    _eps = eps(T)
    @test all(abs(residual) < _eps for residual in residuals)
    @test ev(A, b) == residuals
    @test all(abs(residual) < _eps for residual in ev(A, b))

    return nothing
end


function test_cost_functions(::Type{T}, method::Function, order::Int) where {T}

    A, b = method(T)
    num_stages = length(b)
    @assert size(A) == (num_stages, num_stages)
    @assert size(b) == (num_stages,)
    ev = RKOCEvaluator{T}(order, num_stages)

    l1_cost = L1RKObjective{T}()
    l2_cost = L2RKObjective{T}()
    linf_cost = LInfinityRKObjective{T}()
    huber_cost = HuberRKObjective{T}(one(T))

    if isbitstype(T)
        @test iszero(@allocated ev(l1_cost, A, b))
        @test iszero(@allocated ev(l2_cost, A, b))
        @test iszero(@allocated ev(linf_cost, A, b))
        @test iszero(@allocated ev(huber_cost, A, b))
    end

    _zero = zero(T)
    _eps = eps(T)
    _eps_l1 = _eps * T(length(ev.inv_gamma))
    _eps_2 = _eps * _eps

    @test _zero <= ev(l1_cost, A, b) < _eps_l1
    @test _zero <= ev(l2_cost, A, b) < _eps_2
    @test _zero <= ev(linf_cost, A, b) < _eps
    @test _zero <= ev(huber_cost, A, b) < _eps_2

    for _ = 1:NUM_RANDOM_TRIALS

        weights = rand(T, length(ev.inv_gamma))
        weighted_l1_cost = WeightedL1RKObjective{T}(weights)
        weighted_l2_cost = WeightedL2RKObjective{T}(weights)
        weighted_linf_cost = WeightedLInfinityRKObjective{T}(weights)
        weighted_huber_cost = WeightedHuberRKObjective{T}(one(T), weights)

        if isbitstype(T)
            @test iszero(@allocated ev(weighted_l1_cost, A, b))
            @test iszero(@allocated ev(weighted_l2_cost, A, b))
            @test iszero(@allocated ev(weighted_linf_cost, A, b))
            @test iszero(@allocated ev(weighted_huber_cost, A, b))
        end

        @test _zero <= ev(weighted_l1_cost, A, b) < _eps_l1
        @test _zero <= ev(weighted_l2_cost, A, b) < _eps_2
        @test _zero <= ev(weighted_linf_cost, A, b) < _eps
        @test _zero <= ev(weighted_huber_cost, A, b) < _eps_2

    end

    return nothing
end


using RungeKuttaToolKit.ExampleMethods: RK4, GL6


@testset "residual calculation" begin
    for T in NUMERIC_TYPES
        test_residuals(T, RK4, 4)
        test_residuals(T, GL6, 6)
    end
end


@testset "cost functions" begin
    for T in NUMERIC_TYPES
        test_cost_functions(T, RK4, 4)
        test_cost_functions(T, GL6, 6)
    end
end


######################################################## DIRECTIONAL DERIVATIVES


function test_directional_derivatives(::Type{T}) where {T}

    _eps = eps(T)
    _sqrt_eps = sqrt(_eps)
    _2_sqrt_eps = _sqrt_eps + _sqrt_eps
    _4_sqrt_eps = _2_sqrt_eps + _2_sqrt_eps
    h = _sqrt_eps

    for num_stages = 0:MAX_NUM_STAGES

        trees = random_trees()
        ev = RKOCEvaluator{T}(trees, num_stages)

        A = rand(T, num_stages, num_stages)
        b = rand(T, num_stages)
        dA = rand(T, num_stages, num_stages)
        db = rand(T, num_stages)

        dresiduals_analytic = Vector{T}(undef, length(trees))
        if isbitstype(T)
            @test iszero(@allocated ev'(dresiduals_analytic, A, dA, b, db))
        else
            ev'(dresiduals_analytic, A, dA, b, db)
        end

        residuals_hi = Vector{T}(undef, length(trees))
        residuals_lo = Vector{T}(undef, length(trees))
        ev(residuals_hi, A + h * dA, b + h * db)
        ev(residuals_lo, A - h * dA, b - h * db)
        dresiduals_numerical = (residuals_hi - residuals_lo) / (h + h)

        # num_stages == 1 is pathological.
        if num_stages != 1
            @test maximum_relative_difference(
                dresiduals_analytic, dresiduals_numerical) < _4_sqrt_eps
        end

    end

    return nothing
end


@testset "directional derivatives" begin
    for T in NUMERIC_TYPES
        test_directional_derivatives(T)
    end
end


############################################################ PARTIAL DERIVATIVES


function test_partial_derivatives(::Type{T}) where {T}

    _eps = eps(T)
    _2_eps = _eps + _eps
    _4_eps = _2_eps + _2_eps

    for num_stages = 1:MAX_NUM_STAGES

        trees = random_trees()
        ev = RKOCEvaluator{T}(trees, num_stages)

        A = rand(T, num_stages, num_stages)
        b = rand(T, num_stages)
        i = rand(1:num_stages)
        j = rand(1:num_stages)
        k = rand(1:num_stages)

        dresiduals_fast = Vector{T}(undef, length(trees))
        dresiduals_slow = Vector{T}(undef, length(trees))
        dA = zeros(T, num_stages, num_stages)
        db = zeros(T, num_stages)

        if isbitstype(T)
            @test iszero(@allocated ev'(dresiduals_fast, A, i, j, b))
        else
            ev'(dresiduals_fast, A, i, j, b)
        end

        dA[i, j] = one(T)
        ev'(dresiduals_slow, A, dA, b, db)
        dA[i, j] = zero(T)

        @test maximum_relative_difference(
            dresiduals_fast, dresiduals_slow) < _4_eps

        if isbitstype(T)
            @test iszero(@allocated ev'(dresiduals_fast, A, k))
        else
            ev'(dresiduals_fast, A, k)
        end

        db[k] = one(T)
        ev'(dresiduals_slow, A, dA, b, db)
        db[k] = zero(T)

        @test iszero(maximum_relative_difference(
            dresiduals_fast, dresiduals_slow))

    end

    return nothing
end


@testset "partial derivatives" begin
    for T in NUMERIC_TYPES
        test_partial_derivatives(T)
    end
end


###################################################################### GRADIENTS


# This needs to be a separate function to trigger
# recompilation for each type of cost function.
test_gradient_allocs(ev, gA, gb, obj, A, b) =
    @test iszero(@allocated ev'(gA, gb, obj, A, b))


function test_gradient(::Type{T}) where {T}

    _eps = eps(T)
    _sqrt_eps = sqrt(_eps)
    tolerance = _sqrt_eps
    for _ = 1:8
        tolerance += tolerance
    end
    h = _sqrt_eps

    for num_stages = 0:MAX_NUM_STAGES

        trees = random_trees()
        ev = RKOCEvaluator{T}(trees, num_stages)

        A = rand(T, num_stages, num_stages)
        b = rand(T, num_stages)
        gA = similar(A)
        gb = similar(b)
        dA = zeros(T, num_stages, num_stages)
        db = zeros(T, num_stages)

        l1_cost = L1RKObjective{T}()
        l2_cost = L2RKObjective{T}()
        linf_cost = LInfinityRKObjective{T}()
        huber_cost = HuberRKObjective{T}(one(T))

        weights = rand(T, length(ev.inv_gamma))
        weighted_l1_cost = WeightedL1RKObjective{T}(weights)
        weighted_l2_cost = WeightedL2RKObjective{T}(weights)
        weighted_linf_cost = WeightedLInfinityRKObjective{T}(weights)
        weighted_huber_cost = WeightedHuberRKObjective{T}(one(T), weights)

        for obj in [l2_cost, weighted_l2_cost]

            if isbitstype(T)
                test_gradient_allocs(ev, gA, gb, obj, A, b)
            else
                ev'(gA, gb, obj, A, b)
            end

            if num_stages > 0
                for _ = 1:NUM_RANDOM_TRIALS

                    i = rand(1:num_stages)
                    j = rand(1:num_stages)
                    k = rand(1:num_stages)

                    dA[i, j] = one(T)
                    nA = (ev(obj, A + h * dA, b) -
                          ev(obj, A - h * dA, b)) / (h + h)
                    dA[i, j] = zero(T)
                    db[k] = one(T)
                    nb = (ev(obj, A, b + h * db) -
                          ev(obj, A, b - h * db)) / (h + h)
                    db[k] = zero(T)

                    @test !(relative_difference(gA[i, j], nA) > tolerance)
                    @test !(relative_difference(gb[k], nb) > tolerance)

                end
            end

        end

    end

    return nothing
end


@testset "gradients" begin
    for T in NUMERIC_TYPES
        test_gradient(T)
    end
end
