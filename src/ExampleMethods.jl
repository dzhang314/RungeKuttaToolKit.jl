module ExampleMethods


export RungeKutta4


function RungeKutta4(::Type{T}) where {T}
    _zero = zero(T)
    _one = one(T)
    _two = _one + _one
    _four = _two + _two
    _three = _two + _one
    _six = _four + _two

    _half = inv(_two)
    _third = inv(_three)
    _sixth = inv(_six)

    A = [_zero _zero _zero _zero;
        _half _zero _zero _zero;
        _zero _half _zero _zero;
        _zero _zero _one _zero]
    b = [_sixth, _third, _third, _sixth]
    return (A, b)
end


export GaussLegendre6


function GaussLegendre6(::Type{T}) where {T}
    _1 = one(T)
    _2 = _1 + _1
    _4 = _2 + _2
    _8 = _4 + _4
    _16 = _8 + _8
    _32 = _16 + _16

    _3 = _2 + _1
    _5 = _4 + _1
    _6 = _4 + _2
    _7 = _4 + _3
    _9 = _8 + _1
    _14 = _8 + _6
    _15 = _8 + _7
    _18 = _16 + _2
    _24 = _16 + _8
    _30 = _16 + _14
    _36 = _32 + _4

    _2_9 = _2 / _9
    _4_9 = _4 / _9
    _5_18 = _5 / _18
    _5_36 = _5 / _36

    _sqrt_15 = sqrt(_15)
    _sqrt_15_15 = _sqrt_15 / _15
    _sqrt_15_24 = _sqrt_15 / _24
    _sqrt_15_30 = _sqrt_15 / _30

    A = [(_5_36) (_2_9-_sqrt_15_15) (_5_36-_sqrt_15_30);
        (_5_36+_sqrt_15_24) (_2_9) (_5_36-_sqrt_15_24);
        (_5_36+_sqrt_15_30) (_2_9+_sqrt_15_15) (_5_36)]
    b = [(_5_18), (_4_9), (_5_18)]
    return (A, b)
end


end # module ExampleMethods


# # This is the Runge-Kutta method obtained by applying Richardson extrapolation
# # to n independent executions of Euler's method using step sizes h, h/2, h/3,
# # ..., h/n, yielding a method of order n. The methods obtained in this fashion
# # are not practically useful, but provide a simple proof that Runge-Kutta
# # methods exist of any order using a quadratic number of stages.
# function extrapolated_euler_table(order::Int)::Vector{Rational{BigInt}}
#     result = Vector{Rational{BigInt}}[]
#     skip = 0
#     for i = 2:order
#         for j = 1:i-1
#             push!(result, vcat(
#                 [Rational{BigInt}(1, i)],
#                 zeros(Rational{BigInt}, skip),
#                 [Rational{BigInt}(1, i) for _ = 1:j-1]
#             ))
#         end
#         skip += i - 1
#     end
#     num_stages = div(order * (order - 1), 2) + 1
#     mat = zeros(Rational{BigInt}, order, num_stages)
#     skip = 0
#     for i = 1:order
#         mat[i, 1] = Rational{BigInt}(1, i)
#         for j = skip+2:skip+i
#             mat[i, j] = Rational{BigInt}(1, i)
#         end
#         skip += i - 1
#     end
#     lhs = [Rational{BigInt}(1, j)^i for i = 0:order-1, j = 1:order]
#     rhs = [Rational{BigInt}(i == 1, 1) for i = 1:order]
#     push!(result, transpose(mat) * (lhs \ rhs))
#     vcat(result...)
# end

# function extrapolated_euler_table(
#     ::Type{T}, order::Int
# )::Vector{T} where {T<:Real}
#     T.(extrapolated_euler_table(order))
# end

# # This is the 5th-order method presented on p. 206 of the following paper.
# # Interestingly, it is presented not as a standalone method or in an embedded
# # pair, but as the highest-order component of an embedded quintuple of orders
# # 5(4,3,2,1). Cash and Karp themselves named it "RKFNC"; I'm not sure why.
# # Numerical Recipes calls this method "RKCK."
# # Cash and Karp 1990, "A Variable Order Runge-Kutta Method for Initial Value
# #                      Problems with Rapidly Varying Right-Hand Sides"
# rkck5_table(::Type{T}) where {T<:Real} = T[
#     inv(T(5)),
#     T(3)/T(40), T(9)/T(40),
#     T(3)/T(10), T(-9)/T(10), T(6)/T(5),
#     T(-11)/T(54), T(5)/T(2), T(-70)/T(27), T(35)/T(27),
#     T(1631)/T(55296), T(175)/T(512), T(575)/T(13824), T(44275)/T(110592), T(253)/T(4096),
#     T(37)/T(378), T(0), T(250)/T(621), T(125)/T(594), T(0), T(512)/T(1771)
# ]

# # This is the higher-order component of the 5(4) embedded pair presented on
# # p. 23 of the following paper. Dormand and Prince call this method "RK5(4)7M,"
# # but it has become commonly known as "DOPRI5" following a popular Fortran
# # implementation.
# # Dormand and Prince 1980, "A family of embedded Runge-Kutta formulae"
# dopri5_table(::Type{T}) where {T<:Real} = T[
#     inv(T(5)),
#     T(3)/T(40), T(9)/T(40),
#     T(44)/T(45), T(-56)/T(15), T(32)/T(9),
#     T(19372)/T(6561), T(-25360)/T(2187), T(64448)/T(6561), T(-212)/T(729),
#     T(9017)/T(3168), T(-355)/T(33), T(46732)/T(5247), T(49)/T(176), T(-5103)/T(18656),
#     T(35)/T(384), T(0), T(500)/T(1113), T(125)/T(192), T(-2187)/T(6784), T(11)/T(84),
#     T(35)/T(384), T(0), T(500)/T(1113), T(125)/T(192), T(-2187)/T(6784), T(11)/T(84), T(0)
# ]

# # This is the 8th-order method presented on p. 65 of Fehlberg's 1968 NASA
# # paper as the higher-order component of the embedded pair RK7(8).
# # Fehlberg 1968, "Classical Fifth-, Sixth-, Seventh-, and Eighth-Order
# #                 Runge-Kutta Formulas with Stepsize Control"
# rkf8_table(::Type{T}) where {T<:Real} = T[
#     T(2)/T(27),
#     inv(T(36)), inv(T(12)),
#     inv(T(24)), T(0), inv(T(8)),
#     T(5)/T(12), T(0), T(-25)/T(16), T(25)/T(16),
#     inv(T(20)), T(0), T(0), inv(T(4)), inv(T(5)),
#     T(-25)/T(108), T(0), T(0), T(125)/T(108), T(-65)/T(27), T(125)/T(54),
#     T(31)/T(300), T(0), T(0), T(0), T(61)/T(225), T(-2)/T(9), T(13)/T(900),
#     T(2), T(0), T(0), T(-53)/T(6), T(704)/T(45), T(-107)/T(9), T(67)/T(90), T(3),
#     T(-91)/T(108), T(0), T(0), T(23)/T(108), T(-976)/T(135), T(311)/T(54), T(-19)/T(60), T(17)/T(6), T(-1)/T(12),
#     T(2383)/T(4100), T(0), T(0), T(-341)/T(164), T(4496)/T(1025), T(-301)/T(82), T(2133)/T(4100), T(45)/T(82), T(45)/T(164), T(18)/T(41),
#     T(3)/T(205), T(0), T(0), T(0), T(0), T(-6)/T(41), T(-3)/T(205), T(-3)/T(41), T(3)/T(41), T(6)/T(41), T(0),
#     T(-1777)/T(4100), T(0), T(0), T(-341)/T(164), T(4496)/T(1025), T(-289)/T(82), T(2193)/T(4100), T(51)/T(82), T(33)/T(164), T(12)/T(41), T(0), T(1),
#     T(0), T(0), T(0), T(0), T(0), T(34)/T(105), T(9)/T(35), T(9)/T(35), T(9)/T(280), T(9)/T(280), T(0), T(41)/T(840), T(41)/T(840)
# ]
