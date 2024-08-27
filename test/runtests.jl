using MultiFloats
using RungeKuttaToolKit
using StatsBase: sample
using Test


const NUMERIC_TYPES = [
    Float16, Float32, Float64, BigFloat,
    Float64x1, Float64x2, Float64x3, Float64x4,
    Float64x5, Float64x6, Float64x7, Float64x8,]


function maximum_relative_difference(
    xs::AbstractArray{T},
    ys::AbstractArray{T},
) where {T}
    @assert axes(xs) == axes(ys)
    result = zero(T)
    for (x, y) in zip(xs, ys)
        if !(iszero(x) & iszero(y))
            half_diff = abs(x - y) / (abs(x) + abs(y))
            diff = half_diff + half_diff
            if diff > result
                result = diff
            end
        end
    end
    return result
end


#=


################################################################################


using RungeKuttaToolKit: reshape_explicit!


@testset "reshape_explicit!" begin
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
end


################################################################################


using RungeKuttaToolKit: reshape_diagonally_implicit!


@testset "reshape_diagonally_implicit!" begin
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
end


################################################################################


using RungeKuttaToolKit: reshape_implicit!


@testset "reshape_implicit!" begin
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


################################################################################


using RungeKuttaToolKit.ButcherInstructions: execute_instructions


function test_instructions(::Type{T}, order::Int) where {T}
    ev = RKOCEvaluator{T}(order, 0)
    @test all(i == j for (i, j) in pairs(ev.table.selected_indices))
    @test execute_instructions(ev.table.instructions) == all_rooted_trees(order)
    @test ev.inv_gamma == inv.(T.(butcher_density.(all_rooted_trees(order))))
end


@testset "RKOCEvaluator{T}($order, num_stages)" for order = 0:10
    for T in NUMERIC_TYPES
        test_instructions(T, order)
    end
end


################################################################################


function test_residuals(::Type{T}, method::Function, order::Int) where {T}
    _zero = zero(T)
    _eps = eps(T)
    _eps_2 = _eps * _eps

    A, b = method(T)
    num_stages = length(b)
    @assert size(A) == (num_stages, num_stages)
    @assert size(b) == (num_stages,)
    ev = RKOCEvaluator{T}(order, num_stages)

    if isbitstype(T)
        @test iszero(@allocated ev(A, b))
    end
    @test _zero <= ev(A, b) < _eps_2

    residuals = similar(ev.inv_gamma)
    if isbitstype(T)
        @test iszero(@allocated ev(residuals, A, b))
    else
        ev(residuals, A, b)
    end
    @test all(abs(residual) < _eps for residual in residuals)

    return nothing
end


using RungeKuttaToolKit.ExampleMethods: RK4, GL6


@testset "Residual calculation ($T)" for T in NUMERIC_TYPES
    test_residuals(T, RK4, 4)
    test_residuals(T, GL6, 6)
end


################################################################################


function test_directional_derivatives(::Type{T}) where {T}

    _eps = eps(T)
    _sqrt_eps = sqrt(_eps)
    _2_sqrt_eps = _sqrt_eps + _sqrt_eps
    _4_sqrt_eps = _2_sqrt_eps + _2_sqrt_eps
    h = _sqrt_eps
    all_trees = all_rooted_trees(10)

    for num_stages = 0:20
        trees = sample(all_trees, rand(1:length(all_trees));
            replace=false, ordered=false)
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

        # num_stages == 1 is pathological for some reason.
        if num_stages != 1
            @test maximum_relative_difference(
                dresiduals_analytic, dresiduals_numerical) < _4_sqrt_eps
        end
    end
end


@testset "Directional derivatives ($T)" for T in NUMERIC_TYPES
    test_directional_derivatives(T)
end


################################################################################


function test_partial_derivatives(::Type{T}) where {T}

    _eps = eps(T)
    _2_eps = _eps + _eps
    _4_eps = _2_eps + _2_eps
    all_trees = all_rooted_trees(10)

    for num_stages = 1:20
        trees = sample(all_trees, rand(1:length(all_trees));
            replace=false, ordered=false)
        ev = RKOCEvaluator{T}(trees, num_stages)

        A = rand(T, num_stages, num_stages)
        b = rand(T, num_stages)
        i = rand(1:num_stages)
        j = rand(1:num_stages)

        dresiduals_fast = Vector{T}(undef, length(trees))
        if isbitstype(T)
            @test iszero(@allocated ev'(dresiduals_fast, A, i, j, b))
        else
            ev'(dresiduals_fast, A, i, j, b)
        end

        dA = zeros(T, num_stages, num_stages)
        db = zeros(T, num_stages)
        dA[i, j] = one(T)

        dresiduals_slow = Vector{T}(undef, length(trees))
        ev'(dresiduals_slow, A, dA, b, db)

        @test maximum_relative_difference(
            dresiduals_fast, dresiduals_slow) < _4_eps

        k = rand(1:num_stages)
        if isbitstype(T)
            @test iszero(@allocated ev'(dresiduals_fast, A, k))
        else
            ev'(dresiduals_fast, A, k)
        end

        dA[i, j] = zero(T)
        db[k] = one(T)
        ev'(dresiduals_slow, A, dA, b, db)

        @test iszero(maximum_relative_difference(
            dresiduals_fast, dresiduals_slow))
    end
end


@testset "Partial derivatives ($T)" for T in NUMERIC_TYPES
    test_partial_derivatives(T)
end


=#


################################################################################


function test_gradient(::Type{T}) where {T}

    _eps = eps(T)
    _sqrt_eps = sqrt(_eps)
    tolerance = _sqrt_eps
    for _ = 1:10
        tolerance += tolerance
    end
    h = _sqrt_eps
    all_trees = all_rooted_trees(10)

    for num_stages = 0:8
        trees = sample(all_trees, rand(1:length(all_trees));
            replace=false, ordered=false)
        ev = RKOCEvaluator{T}(trees, num_stages)

        A = rand(T, num_stages, num_stages)
        b = rand(T, num_stages)
        gA_analytic = similar(A)
        gb_analytic = similar(b)

        if isbitstype(T)
            @test iszero(@allocated ev'(gA_analytic, gb_analytic, A, b))
        else
            ev'(gA_analytic, gb_analytic, A, b)
        end

        dA = zeros(T, num_stages, num_stages)
        db = zeros(T, num_stages)
        gA_numerical = similar(A)
        gb_numerical = similar(b)
        for i = 1:num_stages
            for j = 1:num_stages
                dA[i, j] = one(T)
                gA_numerical[i, j] = (
                    ev(A + h * dA, b) - ev(A - h * dA, b)) / (h + h)
                dA[i, j] = zero(T)
            end
        end
        for k = 1:num_stages
            db[k] = one(T)
            gb_numerical[k] = (
                ev(A, b + h * db) - ev(A, b - h * db)) / (h + h)
            db[k] = zero(T)
        end

        @test maximum_relative_difference(
            gA_analytic, gA_numerical) < tolerance
        @test maximum_relative_difference(
            gb_analytic, gb_numerical) < tolerance
    end
end


@testset "Gradient ($T)" for T in NUMERIC_TYPES
    test_gradient(T)
end
