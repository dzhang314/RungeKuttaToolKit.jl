using MultiFloats
using Random: shuffle!
using RungeKuttaToolKit
using StatsBase: sample
using Test


const MAX_ORDER = 10
const MAX_NUM_STAGES = 20
const NUM_RANDOM_TRIALS = 10
const NUM_TOLERANCE_BITS = 10
const NUMERIC_TYPES = [
    Float16, Float32, Float64, BigFloat,
    Float64x1, Float64x2, Float64x3, Float64x4,
    Float64x5, Float64x6, Float64x7, Float64x8]
const FAST_NUMERIC_TYPES = [
    Float16, Float32, Float64,
    Float64x1, Float64x2, Float64x3]


using RungeKuttaToolKit.ButcherInstructions:
    is_palm, butcher_bracket, extract_legs


shuffle_tree(tree::LevelSequence) =
    is_palm(tree) ? copy(tree) : butcher_bracket(
        shuffle!(shuffle_tree.(extract_legs(tree))))


const ALL_TREES = all_rooted_trees(MAX_ORDER)
random_trees() = shuffle_tree.(sample(ALL_TREES, rand(1:length(ALL_TREES));
    replace=false, ordered=false))


function relative_difference(x::T, y::T) where {T}
    x_zero = iszero(x)
    y_zero = iszero(y)
    if x_zero & y_zero
        return zero(T)
    elseif x_zero
        return abs(y)
    elseif y_zero
        return abs(x)
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


######################################################### ROOTED TREE GENERATION


using RungeKuttaToolKit.ButcherInstructions: is_canonical, RevLexIterator


@testset "rooted tree generation" begin

    accumulated_lex = LevelSequence[]
    accumulated_revlex = LevelSequence[]
    accumulated_attach = LevelSequence[]
    accumulated_butcher = LevelSequence[]

    for n = 0:MAX_ORDER

        # Test rooted_trees.

        trees_lex = rooted_trees(n; tree_ordering=:lexicographic)
        trees_revlex = rooted_trees(n; tree_ordering=:reverse_lexicographic)
        trees_attach = rooted_trees(n; tree_ordering=:attach)
        trees_butcher = rooted_trees(n; tree_ordering=:butcher)

        @test allunique(trees_lex)
        @test allunique(trees_revlex)
        @test allunique(trees_attach)
        @test allunique(trees_butcher)

        @test all(length(tree) == n for tree in trees_lex)
        @test all(length(tree) == n for tree in trees_revlex)
        @test all(length(tree) == n for tree in trees_attach)
        @test all(length(tree) == n for tree in trees_butcher)

        @test all(is_canonical, trees_lex)
        @test all(is_canonical, trees_revlex)
        @test all(is_canonical, trees_attach)
        @test all(is_canonical, trees_butcher)

        @test length(trees_lex) == length(RevLexIterator(n))
        @test length(trees_revlex) == length(RevLexIterator(n))
        @test length(trees_attach) == length(RevLexIterator(n))
        @test length(trees_butcher) == length(RevLexIterator(n))

        # Test agreement of different tree orderings.

        @test Set{LevelSequence}(trees_lex) == Set{LevelSequence}(trees_revlex)
        @test Set{LevelSequence}(trees_lex) == Set{LevelSequence}(trees_attach)
        @test Set{LevelSequence}(trees_lex) == Set{LevelSequence}(trees_butcher)

        # Test all_rooted_trees.

        all_trees_lex = all_rooted_trees(n; tree_ordering=:lexicographic)
        all_trees_revlex = all_rooted_trees(n; tree_ordering=:reverse_lexicographic)
        all_trees_attach = all_rooted_trees(n; tree_ordering=:attach)
        all_trees_butcher = all_rooted_trees(n; tree_ordering=:butcher)

        @test allunique(all_trees_lex)
        @test allunique(all_trees_revlex)
        @test allunique(all_trees_attach)
        @test allunique(all_trees_butcher)

        @test all(1 <= length(tree) <= n for tree in all_trees_lex)
        @test all(1 <= length(tree) <= n for tree in all_trees_revlex)
        @test all(1 <= length(tree) <= n for tree in all_trees_attach)
        @test all(1 <= length(tree) <= n for tree in all_trees_butcher)

        @test all(is_canonical, all_trees_lex)
        @test all(is_canonical, all_trees_revlex)
        @test all(is_canonical, all_trees_attach)
        @test all(is_canonical, all_trees_butcher)

        @test issorted(all_trees_lex, by=length)
        @test issorted(all_trees_revlex, by=length)
        @test issorted(all_trees_attach, by=length)
        @test issorted(all_trees_butcher, by=length)

        # Test agreement between rooted_trees and all_rooted_trees.

        append!(accumulated_lex, trees_lex)
        append!(accumulated_revlex, trees_revlex)
        append!(accumulated_attach, trees_attach)
        append!(accumulated_butcher, trees_butcher)

        @test accumulated_lex == all_trees_lex
        @test accumulated_revlex == all_trees_revlex
        @test accumulated_attach == all_trees_attach
        @test accumulated_butcher == all_trees_butcher

    end
end


###################################################### ROOTED TREE COMBINATORICS


using RungeKuttaToolKit.ButcherInstructions:
    attach_leaf_left, attach_leaf_right, is_canonical, canonize, counts


@testset "rooted tree combinatorics" begin
    let
        left = [LevelSequence([1])]
        right = [LevelSequence([1])]
        for order = 2:MAX_ORDER
            left = [attach_leaf_left(tree, i)
                    for tree in left for i = 1:length(tree)]
            right = [attach_leaf_right(tree, i)
                     for tree in right for i = 1:length(tree)]
            if order > 3
                @test !all(is_canonical, left)
                @test !all(is_canonical, right)
            end
            left_canon = canonize.(left)
            right_canon = canonize.(right)
            @test all(is_canonical, left_canon)
            @test all(is_canonical, right_canon)
            left_counts = counts(left_canon)
            right_counts = counts(right_canon)
            @test left_counts == right_counts
            for (tree, alpha) in left_counts
                @test (factorial(BigInt(order)) ==
                       alpha * butcher_density(tree) * butcher_symmetry(tree))
            end
        end
    end
end


############################################################## RESHAPE OPERATORS


using RungeKuttaToolKit.RKParameterization


@testset "reshape operators" begin
    let
        A, b = RKParameterizationExplicit{BigFloat}(3)(
            Matrix{BigFloat}(undef, 3, 3),
            Vector{BigFloat}(undef, 3),
            BigFloat.(1:6))
        @test A == BigFloat[0 0 0; 1 0 0; 2 3 0]
        @test b == BigFloat.(4:6)
    end
    let
        x = RKParameterizationExplicit{BigFloat}(3)(
            Vector{BigFloat}(undef, 6),
            BigFloat[0 0 0; 1 0 0; 2 3 0],
            BigFloat.(4:6))
        @test x == BigFloat.(1:6)
    end
    let
        A = RKParameterizationExplicitAO{BigFloat}(3)(
            Matrix{BigFloat}(undef, 3, 3),
            BigFloat.(1:3))
        @test A == BigFloat[0 0 0; 1 0 0; 2 3 0]
    end
    let
        x = RKParameterizationExplicitAO{BigFloat}(3)(
            Vector{BigFloat}(undef, 3),
            BigFloat[0 0 0; 1 0 0; 2 3 0])
        @test x == BigFloat.(1:3)
    end
    let
        A, b = RKParameterizationDiagonallyImplicit{BigFloat}(3)(
            Matrix{BigFloat}(undef, 3, 3),
            Vector{BigFloat}(undef, 3),
            BigFloat.(1:9))
        @test A == BigFloat[1 0 0; 2 3 0; 4 5 6]
        @test b == BigFloat.(7:9)
    end
    let
        x = RKParameterizationDiagonallyImplicit{BigFloat}(3)(
            Vector{BigFloat}(undef, 9),
            BigFloat[1 0 0; 2 3 0; 4 5 6],
            BigFloat.(7:9))
        @test x == BigFloat.(1:9)
    end
    let
        A = RKParameterizationDiagonallyImplicitAO{BigFloat}(3)(
            Matrix{BigFloat}(undef, 3, 3),
            BigFloat.(1:6))
        @test A == BigFloat[1 0 0; 2 3 0; 4 5 6]
    end
    let
        x = RKParameterizationDiagonallyImplicitAO{BigFloat}(3)(
            Vector{BigFloat}(undef, 6),
            BigFloat[1 0 0; 2 3 0; 4 5 6])
        @test x == BigFloat.(1:6)
    end
    let
        A, b = RKParameterizationImplicit{BigFloat}(3)(
            Matrix{BigFloat}(undef, 3, 3),
            Vector{BigFloat}(undef, 3),
            BigFloat.(1:12))
        @test A == BigFloat[1 2 3; 4 5 6; 7 8 9]
        @test b == BigFloat.(10:12)
    end
    let
        x = RKParameterizationImplicit{BigFloat}(3)(
            Vector{BigFloat}(undef, 12),
            BigFloat[1 2 3; 4 5 6; 7 8 9],
            BigFloat.(10:12))
        @test x == BigFloat.(1:12)
    end
    let
        A = RKParameterizationImplicitAO{BigFloat}(3)(
            Matrix{BigFloat}(undef, 3, 3),
            BigFloat.(1:9))
        @test A == BigFloat[1 2 3; 4 5 6; 7 8 9]
    end
    let
        x = RKParameterizationImplicitAO{BigFloat}(3)(
            Vector{BigFloat}(undef, 9),
            BigFloat[1 2 3; 4 5 6; 7 8 9])
        @test x == BigFloat.(1:9)
    end
    let
        A, b = RKParameterizationParallelExplicit{BigFloat}(3, 2)(
            Matrix{BigFloat}(undef, 7, 7),
            Vector{BigFloat}(undef, 7),
            BigFloat.(1:25))
        @test A == BigFloat[
            00 00 00 00 00 0 0;
            01 00 00 00 00 0 0;
            02 00 00 00 00 0 0;
            03 04 05 00 00 0 0;
            06 07 08 00 00 0 0;
            09 10 11 12 13 0 0;
            14 15 16 17 18 0 0]
        @test b == BigFloat.(19:25)
    end
    let
        x = RKParameterizationParallelExplicit{BigFloat}(3, 2)(
            Vector{BigFloat}(undef, 25),
            BigFloat[
                00 00 00 00 00 0 0;
                01 00 00 00 00 0 0;
                02 00 00 00 00 0 0;
                03 04 05 00 00 0 0;
                06 07 08 00 00 0 0;
                09 10 11 12 13 0 0;
                14 15 16 17 18 0 0],
            BigFloat.(19:25))
        @test x == BigFloat.(1:25)
    end
end


##################################################### RKOCEVALUATOR CONSTRUCTION


using RungeKuttaToolKit.ButcherInstructions: canonize, execute_instructions


@testset "RKOCEvaluator construction" begin
    for order = 0:MAX_ORDER
        for T in NUMERIC_TYPES
            trees = random_trees()
            ev = RKOCEvaluator{T}(trees, order)
            computed = execute_instructions(ev.table.instructions)
            @test canonize.(computed[ev.table.selected_indices]) == canonize.(trees)
            @test ev.inv_gamma == inv.(T.(butcher_density.(trees)))
        end
    end
end


###################################################################### RESIDUALS


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


using RungeKuttaToolKit.ExampleMethods: RK4, GL6


@testset "residual calculation" begin
    for T in NUMERIC_TYPES
        test_residuals(T, RK4, 4)
        test_residuals(T, GL6, 6)
    end
end


################################################################# COST FUNCTIONS


using RungeKuttaToolKit.RKCost


function test_cost_functions(::Type{T}, method::Function, order::Int) where {T}

    A, b = method(T)
    num_stages = length(b)
    @assert size(A) == (num_stages, num_stages)
    @assert size(b) == (num_stages,)
    ev = RKOCEvaluator{T}(order, num_stages)

    l1_cost = RKCostL1{T}()
    l2_cost = RKCostL2{T}()
    linf_cost = RKCostLInfinity{T}()
    huber_cost = RKCostHuber{T}(one(T))

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
        weighted_l1_cost = RKCostWeightedL1{T}(weights)
        weighted_l2_cost = RKCostWeightedL2{T}(weights)
        weighted_linf_cost = RKCostWeightedLInfinity{T}(weights)
        weighted_huber_cost = RKCostWeightedHuber{T}(one(T), weights)

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
test_gradient_allocs(ev, gA, gb, cost, A, b) =
    @test iszero(@allocated ev'(gA, gb, cost, A, b))


function test_gradient(::Type{T}) where {T}

    _eps = eps(T)
    _sqrt_eps = sqrt(_eps)
    tolerance = _sqrt_eps
    for _ = 1:NUM_TOLERANCE_BITS
        tolerance += tolerance
    end
    h = _sqrt_eps

    num_trials = 0
    num_successes = 0

    for num_stages = 0:MAX_NUM_STAGES

        trees = random_trees()
        ev = RKOCEvaluator{T}(trees, num_stages)

        A = rand(T, num_stages, num_stages)
        b = rand(T, num_stages)
        gA = similar(A)
        gb = similar(b)
        dA = zeros(T, num_stages, num_stages)
        db = zeros(T, num_stages)
        delta = rand(T)
        weights = rand(T, length(ev.inv_gamma))

        l1_cost = RKCostL1{T}()
        l2_cost = RKCostL2{T}()
        linf_cost = RKCostLInfinity{T}()
        huber_cost = RKCostHuber{T}(delta)

        weighted_l1_cost = RKCostWeightedL1{T}(weights)
        weighted_l2_cost = RKCostWeightedL2{T}(weights)
        weighted_linf_cost = RKCostWeightedLInfinity{T}(weights)
        weighted_huber_cost = RKCostWeightedHuber{T}(delta, weights)

        for cost in [
            l1_cost, l2_cost, weighted_l1_cost, weighted_l2_cost,
            huber_cost, weighted_huber_cost]

            if isbitstype(T)
                test_gradient_allocs(ev, gA, gb, cost, A, b)
            else
                ev'(gA, gb, cost, A, b)
            end

            if num_stages > 0
                for _ = 1:NUM_RANDOM_TRIALS

                    i = rand(1:num_stages)
                    j = rand(1:num_stages)
                    k = rand(1:num_stages)

                    dA[i, j] = one(T)
                    nA = (ev(cost, A + h * dA, b) -
                          ev(cost, A - h * dA, b)) / (h + h)
                    dA[i, j] = zero(T)
                    db[k] = one(T)
                    nb = (ev(cost, A, b + h * db) -
                          ev(cost, A, b - h * db)) / (h + h)
                    db[k] = zero(T)

                    num_trials += 2
                    num_successes += !(relative_difference(gA[i, j], nA) > tolerance)
                    num_successes += !(relative_difference(gb[k], nb) > tolerance)
                end
            end
        end
    end

    # println(T, " : ", num_successes / num_trials)
    @test num_successes / num_trials > 0.95
    return nothing
end


@testset "gradients" begin
    for T in [Float16, Float32, Float64, Float64x1, Float64x2]
        test_gradient(T)
    end
end


###################################################################### JACOBIANS


function test_jacobian(param::AbstractRKParameterization{T}) where {T}

    _eps = eps(T)
    _sqrt_eps = sqrt(_eps)
    tolerance = _sqrt_eps
    for _ = 1:NUM_TOLERANCE_BITS
        tolerance += tolerance
    end
    h = _sqrt_eps

    trees = random_trees()
    ev = RKOCEvaluator{T}(trees, param.num_stages)
    x = rand(T, param.num_variables)
    dx = rand(T, param.num_variables)
    A = Matrix{T}(undef, param.num_stages, param.num_stages)
    dA = Matrix{T}(undef, param.num_stages, param.num_stages)
    b = Vector{T}(undef, param.num_stages)
    db = Vector{T}(undef, param.num_stages)
    param(A, b, x)
    param(dA, db, dx)

    dresiduals = (ev(A + h * dA, b + h * db) -
                  ev(A - h * dA, b - h * db)) / (h + h)

    jacobian = Matrix{T}(undef, length(dresiduals), param.num_variables)
    if isbitstype(T)
        @test iszero(@allocated ev'(jacobian, A, b, param, x))
    else
        ev'(jacobian, A, b, param, x)
    end

    return maximum_relative_difference(dresiduals, jacobian * dx) < tolerance
end


@testset "jacobians" begin
    for T in FAST_NUMERIC_TYPES
        num_trials = 0
        num_successes = 0
        for num_stages = 0:MAX_NUM_STAGES
            num_trials += 5
            num_successes += test_jacobian(RKParameterizationExplicit{T}(num_stages))
            num_successes += test_jacobian(RKParameterizationDiagonallyImplicit{T}(num_stages))
            num_successes += test_jacobian(RKParameterizationImplicit{T}(num_stages))
            num_successes += test_jacobian(RKParameterizationParallelExplicit{T}(num_stages, 1))
            num_successes += test_jacobian(RKParameterizationParallelExplicit{T}(num_stages, 2))
        end
        # println(T, " : ", num_successes / num_trials)
        @test num_successes / num_trials > 0.95
    end
end
