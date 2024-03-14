using Test


################################################################################


using RungeKuttaToolKit: reshape_explicit!


@testset "reshape_explicit!(A, b, x)" begin
    A, b = reshape_explicit!(
        Matrix{BigFloat}(undef, 3, 3),
        Vector{BigFloat}(undef, 3),
        BigFloat[1, 2, 3, 4, 5, 6])
    @test A == BigFloat[0 0 0; 1 0 0; 2 3 0]
    @test b == BigFloat[4, 5, 6]
end


@testset "reshape_explicit!(x, A, b)" begin
    x = reshape_explicit!(
        Vector{BigFloat}(undef, 6),
        BigFloat[0 0 0; 1 0 0; 2 3 0],
        BigFloat[4, 5, 6])
    @test x == BigFloat[1, 2, 3, 4, 5, 6]
end


@testset "reshape_explicit!(A, x)" begin
    A = reshape_explicit!(
        Matrix{BigFloat}(undef, 3, 3),
        BigFloat[1, 2, 3])
    @test A == BigFloat[0 0 0; 1 0 0; 2 3 0]
end


@testset "reshape_explicit!(x, A)" begin
    x = reshape_explicit!(
        Vector{BigFloat}(undef, 3),
        BigFloat[0 0 0; 1 0 0; 2 3 0])
    @test x == BigFloat[1, 2, 3]
end


################################################################################


using RungeKuttaToolKit: reshape_implicit!


@testset "reshape_implicit!(A, b, x)" begin
    A, b = reshape_implicit!(
        Matrix{BigFloat}(undef, 3, 3),
        Vector{BigFloat}(undef, 3),
        BigFloat[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
    @test A == BigFloat[1 2 3; 4 5 6; 7 8 9]
    @test b == BigFloat[10, 11, 12]
end


@testset "reshape_implicit!(x, A, b)" begin
    x = reshape_implicit!(
        Vector{BigFloat}(undef, 12),
        BigFloat[1 2 3; 4 5 6; 7 8 9],
        BigFloat[10, 11, 12])
    @test x == BigFloat[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
end


@testset "reshape_implicit!(A, x)" begin
    A = reshape_implicit!(
        Matrix{BigFloat}(undef, 3, 3),
        BigFloat[1, 2, 3, 4, 5, 6, 7, 8, 9])
    @test A == BigFloat[1 2 3; 4 5 6; 7 8 9]
end


@testset "reshape_implicit!(x, A)" begin
    x = reshape_implicit!(
        Vector{BigFloat}(undef, 9),
        BigFloat[1 2 3; 4 5 6; 7 8 9])
    @test x == BigFloat[1, 2, 3, 4, 5, 6, 7, 8, 9]
end


################################################################################


using RungeKuttaToolKit: RKOCEvaluatorAE, RKOCEvaluatorBE
using RungeKuttaToolKit: RKOCEvaluatorAI, RKOCEvaluatorBI
using RungeKuttaToolKit: RKOCResidualEvaluatorAE, RKOCResidualEvaluatorBE


function test_evaluators_rk4(::Type{T}) where {T}

    _zero = zero(T)
    _one = one(T)
    _two = _one + _one
    _half = inv(_two)
    _three = _two + _one
    _third = inv(_three)
    _four = _two + _two
    _fourth = inv(_four)
    _six = _three + _three
    _sixth = inv(_six)
    _eight = _four + _four
    _eighth = inv(_eight)

    x_ae = [_half, _zero, _half, _zero, _zero, _one]
    x_be = [_half, _zero, _half, _zero, _zero, _one,
        _sixth, _third, _third, _sixth]
    x_ai = [_zero, _zero, _zero, _zero, _half, _zero, _zero, _zero,
        _zero, _half, _zero, _zero, _zero, _zero, _one, _zero]
    x_bi = [_zero, _zero, _zero, _zero, _half, _zero, _zero, _zero,
        _zero, _half, _zero, _zero, _zero, _zero, _one, _zero,
        _sixth, _third, _third, _sixth]

    ev_ae = RKOCEvaluatorAE{T}(4, 4)
    ev_be = RKOCEvaluatorBE{T}(4, 4)
    ev_ai = RKOCEvaluatorAI{T}(4, 4)
    ev_bi = RKOCEvaluatorBI{T}(4, 4)

    _eps = eps(T)
    _eps_squared = _eps * _eps

    @test abs(ev_ae(x_ae)) < _eps_squared
    @test abs(ev_be(x_be)) < _eps_squared
    @test abs(ev_ai(x_ai)) < _eps_squared
    @test abs(ev_bi(x_bi)) < _eps_squared

    phi = [_one _zero _zero _zero _zero _zero _zero _zero;
        _one _half _zero _fourth _zero _zero _zero _eighth;
        _one _half _fourth _fourth _zero _eighth _eighth _eighth;
        _one _one _half _one _fourth _fourth _half _one]

    @test phi == ev_ae.phi
    @test phi == ev_be.phi
    @test phi == ev_ai.phi
    @test phi == ev_bi.phi

    @test all(abs(g) < _eps for g in ev_ae'(x_ae))
    @test all(abs(g) < _eps + _eps for g in ev_be'(x_be))
    @test all(abs(g) < _eps for g in ev_ai'(x_ai))
    @test all(abs(g) < _eps + _eps for g in ev_bi'(x_bi))

    @test abs(ev_ae.b[1] - _sixth) < _eps
    @test abs(ev_ae.b[2] - _third) < _eps
    @test abs(ev_ae.b[3] - _third) < _eps
    @test abs(ev_ae.b[4] - _sixth) < _eps

    @test abs(ev_ai.b[1] - _sixth) < _eps
    @test abs(ev_ai.b[2] - _third) < _eps
    @test abs(ev_ai.b[3] - _third) < _eps
    @test abs(ev_ai.b[4] - _sixth) < _eps

end


@testset "RKOC Evaluators (Float16)" test_evaluators_rk4(Float16)
@testset "RKOC Evaluators (Float32)" test_evaluators_rk4(Float32)
@testset "RKOC Evaluators (Float64)" test_evaluators_rk4(Float64)
@testset "RKOC Evaluators (BigFloat)" test_evaluators_rk4(BigFloat)


################################################################################


using MultiFloats: Float64x1, Float64x2, Float64x3, Float64x4
using MultiFloats: Float64x5, Float64x6, Float64x7, Float64x8


@testset "RKOC Evaluators (Float64x1)" test_evaluators_rk4(Float64x1)
@testset "RKOC Evaluators (Float64x2)" test_evaluators_rk4(Float64x2)
@testset "RKOC Evaluators (Float64x3)" test_evaluators_rk4(Float64x3)
@testset "RKOC Evaluators (Float64x4)" test_evaluators_rk4(Float64x4)
@testset "RKOC Evaluators (Float64x5)" test_evaluators_rk4(Float64x5)
@testset "RKOC Evaluators (Float64x6)" test_evaluators_rk4(Float64x6)
@testset "RKOC Evaluators (Float64x7)" test_evaluators_rk4(Float64x7)
@testset "RKOC Evaluators (Float64x8)" test_evaluators_rk4(Float64x8)
