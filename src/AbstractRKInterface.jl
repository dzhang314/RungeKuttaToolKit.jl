module AbstractRKInterface


using ..RungeKuttaToolKit: AbstractRKOCEvaluator, AbstractRKOCAdjoint,
    AbstractRKCost, AbstractRKParameterization,
    get_axes, compute_Phi!, compute_residuals!,
    pushforward_dPhi!, pushforward_dresiduals!, pullback_dPhi!, pullback_dA!


"""
    (ev::AbstractRKOCEvaluator{T})(
        residuals::AbstractVector{T},
        A::AbstractMatrix{T},
        b::AbstractVector{T},
    ) -> AbstractVector{T}

Compute residuals of the Runge--Kutta order conditions
``\\{ \\mathbf{b} \\cdot \\Phi_t(A) - 1/\\gamma(t) : t \\in T \\}``
for a given Butcher tableau ``(A, \\mathbf{b})``
over a set of rooted trees ``T`` encoded by an `AbstractRKOCEvaluator`.

# Arguments
- `ev`: `AbstractRKOCEvaluator` object encoding a set of rooted trees.
- `residuals`: length ``|T|`` output vector. Each residual
    ``\\mathbf{b} \\cdot \\Phi_t(A) - 1/\\gamma(t)`` is written to
    `residuals[i]` in the order specified when constructing `ev`.
- `A`: ``s \\times s`` input matrix containing the coefficients of a
    Runge--Kutta method (i.e., the upper-right block of a Butcher tableau).
- `b`: length ``s`` input vector containing the weights of a Runge--Kutta
    method (i.e., the lower-right row of a Butcher tableau).

Here, ``|T|`` denotes the number of rooted trees encoded by `ev`, and ``s``
denotes the number of stages specified when constructing `ev`.
"""
function (ev::AbstractRKOCEvaluator{T})(
    residuals::AbstractVector{T},
    A::AbstractMatrix{T},
    b::AbstractVector{T},
) where {T}

    # Validate array dimensions.
    stage_axis, _, output_axis = get_axes(ev)
    @assert axes(residuals) == (output_axis,)
    @assert axes(A) == (stage_axis, stage_axis)
    @assert axes(b) == (stage_axis,)

    compute_Phi!(ev, A)
    return compute_residuals!(residuals, ev, b)
end


function (ev::AbstractRKOCEvaluator{T})(
    A::AbstractMatrix{T},
    b::AbstractVector{T},
) where {T}

    # Validate array dimensions.
    stage_axis, _, output_axis = get_axes(ev)
    @assert axes(A) == (stage_axis, stage_axis)
    @assert axes(b) == (stage_axis,)

    return ev(similar(Vector{T}, output_axis), A, b)
end


"""
    (ev::AbstractRKOCEvaluator{T})(
        cost::AbstractRKCost{T},
        A::AbstractMatrix{T},
        b::AbstractVector{T},
    ) -> T

Compute a specified cost function of the
residuals of the Runge--Kutta order conditions
``\\{ \\mathbf{b} \\cdot \\Phi_t(A) - 1/\\gamma(t) : t \\in T \\}``
for a given Butcher tableau ``(A, \\mathbf{b})``
over a set of rooted trees ``T`` encoded by an `AbstractRKOCEvaluator`.

# Arguments
- `ev`: `AbstractRKOCEvaluator` object encoding a set of rooted trees.
- `cost`: `AbstractRKCost` object specifying the cost function to evaluate.
- `A`: ``s \\times s`` input matrix containing the coefficients of a
    Runge--Kutta method (i.e., the upper-right block of a Butcher tableau).
- `b`: length ``s`` input vector containing the weights of a Runge--Kutta
    method (i.e., the lower-right row of a Butcher tableau).

Here, ``s`` denotes the number of stages specified when constructing `ev`.
"""
function (ev::AbstractRKOCEvaluator{T})(
    cost::AbstractRKCost{T},
    A::AbstractMatrix{T},
    b::AbstractVector{T},
) where {T}

    # Validate array dimensions.
    stage_axis, _, _ = get_axes(ev)
    @assert axes(A) == (stage_axis, stage_axis)
    @assert axes(b) == (stage_axis,)

    compute_Phi!(ev, A)
    return cost(ev, b)
end


"""
    (adj::AbstractRKOCAdjoint{T})(
        dresiduals::AbstractVector{T},
        A::AbstractMatrix{T},
        dA::AbstractMatrix{T},
        b::AbstractVector{T},
        db::AbstractVector{T},
    ) -> AbstractVector{T}

Compute directional derivatives of the Runge--Kutta order conditions
``\\{ \\nabla_{\\mathrm{d}A, \\mathrm{d}\\mathbf{b}} [
\\mathbf{b} \\cdot \\Phi_t(A) ] : t \\in T \\}``
at a given Butcher tableau ``(A, \\mathbf{b})``
in direction ``(\\mathrm{d}A, \\mathrm{d}\\mathbf{b})``
over a set of rooted trees ``T`` encoded by an `AbstractRKOCEvaluator`.

# Arguments
- `adj`: `AbstractRKOCAdjoint` object obtained by applying the adjoint
    operator `'` to an `AbstractRKOCEvaluator`. In other words, this function
    should be called as `ev'(dresiduals, A, dA, b, db)` where `ev` is an
    `AbstractRKOCEvaluator`.
- `dresiduals`: length ``|T|`` output vector. Each directional derivative
    ``\\nabla_{\\mathrm{d}A, \\mathrm{d}\\mathbf{b}} [
    \\mathbf{b} \\cdot \\Phi_t(A) ]`` is written to `dresiduals[i]` in the
    order specified when constructing `ev`.
- `A`: ``s \\times s`` input matrix containing the coefficients of a
    Runge--Kutta method (i.e., the upper-right block of a Butcher tableau).
- `dA`: ``s \\times s`` input matrix containing the direction in which to
    differentiate ``A``.
- `b`: length ``s`` input vector containing the weights of a Runge--Kutta
    method (i.e., the lower-right row of a Butcher tableau).
- `db`: length ``s`` input vector containing the direction in which to
    differentiate ``\\mathbf{b}``.

Here, ``|T|`` denotes the number of rooted trees encoded by `ev`, and ``s``
denotes the number of stages specified when constructing `ev`.
"""
function (adj::AbstractRKOCAdjoint{T})(
    dresiduals::AbstractVector{T},
    A::AbstractMatrix{T},
    dA::AbstractMatrix{T},
    b::AbstractVector{T},
    db::AbstractVector{T},
) where {T}

    # Validate array dimensions.
    stage_axis, _, output_axis = get_axes(adj.ev)
    @assert axes(dresiduals) == (output_axis,)
    @assert axes(A) == (stage_axis, stage_axis)
    @assert axes(dA) == (stage_axis, stage_axis)
    @assert axes(b) == (stage_axis,)
    @assert axes(db) == (stage_axis,)

    compute_Phi!(adj.ev, A)
    pushforward_dPhi!(adj.ev, A, dA)
    return pushforward_dresiduals!(dresiduals, adj.ev, b, db)
end


"""
    (adj::AbstractRKOCAdjoint{T})(
        dresiduals::AbstractVector{T},
        A::AbstractMatrix{T},
        i::Integer,
        j::Integer,
        b::AbstractVector{T},
    ) -> AbstractVector{T}

Compute partial derivatives of the Runge--Kutta order conditions
``\\{ \\partial_{A_{i,j}} [ \\mathbf{b} \\cdot \\Phi_t(A) ] : t \\in T \\}``
with respect to ``A_{i,j}`` at a given Butcher tableau ``(A, \\mathbf{b})``
over a set of rooted trees ``T`` encoded by an `AbstractRKOCEvaluator`.

# Arguments
- `adj`: `AbstractRKOCAdjoint` object obtained by applying the adjoint
    operator `'` to an `AbstractRKOCEvaluator`. In other words, this function
    should be called as `ev'(dresiduals, A, dA, b, db)` where `ev` is an
    `AbstractRKOCEvaluator`.
- `dresiduals`: length ``|T|`` output vector. Each partial derivative
    ``\\partial_{A_{i,j}} [ \\mathbf{b} \\cdot \\Phi_t(A) ]`` is written to
    `dresiduals[i]` in the order specified when constructing `ev`.
- `A`: ``s \\times s`` input matrix containing the coefficients of a
    Runge--Kutta method (i.e., the upper-right block of a Butcher tableau).
- `i`: row index of the entry of ``A`` to differentiate with respect to.
- `j`: column index of the entry of ``A`` to differentiate with respect to.
- `b`: length ``s`` input vector containing the weights of a Runge--Kutta
    method (i.e., the lower-right row of a Butcher tableau).

Here, ``|T|`` denotes the number of rooted trees encoded by `ev`, and ``s``
denotes the number of stages specified when constructing `ev`.

This method uses an optimized algorithm that should be faster than
`ev'(dresiduals, A, dA, b, db)` when differentiating with respect to a single
entry of ``A``. It may produce slightly different results due to the
non-associative nature of floating-point arithmetic.
"""
function (adj::AbstractRKOCAdjoint{T})(
    dresiduals::AbstractVector{T},
    A::AbstractMatrix{T},
    i::Integer,
    j::Integer,
    b::AbstractVector{T},
) where {T}

    # Validate array dimensions.
    stage_axis, _, output_axis = get_axes(adj.ev)
    @assert axes(dresiduals) == (output_axis,)
    @assert axes(A) == (stage_axis, stage_axis)
    @assert i in stage_axis
    @assert j in stage_axis
    @assert axes(b) == (stage_axis,)

    compute_Phi!(adj.ev, A)
    pushforward_dPhi!(adj.ev, A, i, j)
    return pushforward_dresiduals!(dresiduals, adj.ev, b)
end


"""
    (adj::AbstractRKOCAdjoint{T})(
        dresiduals::AbstractVector{T},
        A::AbstractMatrix{T},
        i::Integer,
    ) -> AbstractVector{T}

Compute partial derivatives of the Runge--Kutta order conditions
``\\{ \\partial_{b_i} [ \\mathbf{b} \\cdot \\Phi_t(A) ] : t \\in T \\}``
with respect to ``b_i`` at a given Butcher tableau ``A``
over a set of rooted trees ``T`` encoded by an `AbstractRKOCEvaluator`.
(The result is independent of the value of ``\\mathbf{b}``.)

# Arguments
- `adj`: `AbstractRKOCAdjoint` object obtained by applying the adjoint
    operator `'` to an `AbstractRKOCEvaluator`. In other words, this function
    should be called as `ev'(dresiduals, A, dA, b, db)` where `ev` is an
    `AbstractRKOCEvaluator`.
- `dresiduals`: length ``|T|`` output vector. Each partial derivative
    ``\\partial_{b_i} [ \\mathbf{b} \\cdot \\Phi_t(A) ]`` is written to
    `dresiduals[i]` in the order specified when constructing `ev`.
- `A`: ``s \\times s`` input matrix containing the coefficients of a
    Runge--Kutta method (i.e., the upper-right block of a Butcher tableau).
- `i`: index of the entry of ``\\mathbf{b}`` to differentiate with respect to.

Here, ``|T|`` denotes the number of rooted trees encoded by `ev`, and ``s``
denotes the number of stages specified when constructing `ev`.

This method uses an optimized algorithm that should be faster than
`ev'(dresiduals, A, dA, b, db)` when differentiating with respect to a single
entry of ``\\mathbf{b}``. It may produce slightly different results due to the
non-associative nature of floating-point arithmetic.
"""
function (adj::AbstractRKOCAdjoint{T})(
    dresiduals::AbstractVector{T},
    A::AbstractMatrix{T},
    i::Integer,
) where {T}

    # Validate array dimensions.
    stage_axis, _, output_axis = get_axes(adj.ev)
    @assert axes(dresiduals) == (output_axis,)
    @assert axes(A) == (stage_axis, stage_axis)
    @assert i in stage_axis

    # Compute Butcher weight vectors.
    compute_Phi!(adj.ev, A)

    # Extract entries of Phi.
    @inbounds for (j, k) in pairs(adj.ev.table.selected_indices)
        dresiduals[j] = adj.ev.Phi[i, k]
    end

    return dresiduals
end


# """
#     (adj::RKOCAdjoint{T})(
#         dA::AbstractMatrix{T},
#         db::AbstractVector{T},
#         A::AbstractMatrix{T},
#         b::AbstractVector{T},
#     ) -> Tuple{AbstractMatrix{T}, AbstractVector{T}}

# Compute the gradient of the sum of squared residuals
# of the Runge--Kutta order conditions ``\\nabla_{A, \\mathbf{b}}
# \\sum_{t \\in T} (\\mathbf{b} \\cdot \\Phi_t(A) - 1/\\gamma(t))^2``
# at a given Butcher tableau ``(A, \\mathbf{b})``
# over a set of rooted trees ``T`` encoded by an `RKOCEvaluator`.

# # Arguments
# - `adj`: `RKOCAdjoint` object obtained by applying the adjoint operator `'`
#     to an `RKOCEvaluator`. In other words, this function should be called as
#     `ev'(dA, db, A, b)` where `ev` is an `RKOCEvaluator`.
# - `dA`: ``s \\times s`` output matrix containing the gradient of the sum of
#     squared residuals with respect to ``A``.
# - `db`: length ``s`` output vector containing the gradient of the sum of
#     squared residuals with respect to ``\\mathbf{b}``.
# - `A`: ``s \\times s`` input matrix containing the coefficients of a
#     Runge--Kutta method (i.e., the upper-right block of a Butcher tableau).
# - `b`: length ``s`` input vector containing the weights of a Runge--Kutta
#     method (i.e., the lower-right row of a Butcher tableau).

# Here, ``s`` denotes the number of stages specified when constructing `ev`.
# """
function (adj::AbstractRKOCAdjoint{T})(
    dA::AbstractMatrix{T},
    db::AbstractVector{T},
    cost::AbstractRKCost{T},
    A::AbstractMatrix{T},
    b::AbstractVector{T},
) where {T}

    # Validate array dimensions.
    stage_axis, _, _ = get_axes(adj.ev)
    @assert axes(dA) == (stage_axis, stage_axis)
    @assert axes(db) == (stage_axis,)
    @assert axes(A) == (stage_axis, stage_axis)
    @assert axes(b) == (stage_axis,)

    compute_Phi!(adj.ev, A)
    cost(adj, b)
    pullback_dPhi!(adj.ev, A)
    pullback_dA!(dA, adj.ev)
    cost(db, adj, b)
    return (dA, db)
end


function (adj::AbstractRKOCAdjoint{T})(
    cost::AbstractRKCost{T},
    A::AbstractMatrix{T},
    b::AbstractVector{T},
) where {T}

    # Validate array dimensions.
    stage_axis, _, _ = get_axes(adj.ev)
    @assert axes(A) == (stage_axis, stage_axis)
    @assert axes(b) == (stage_axis,)

    return adj(similar(Matrix{T}, stage_axis, stage_axis),
        similar(Vector{T}, stage_axis), cost, A, b)
end


function (adj::AbstractRKOCAdjoint{T})(
    jacobian::AbstractMatrix{T},
    A::AbstractMatrix{T},
    b::AbstractVector{T},
    param::AbstractRKParameterization{T},
    x::AbstractVector{T},
) where {T}

    # Validate array dimensions.
    stage_axis, _, output_axis = get_axes(adj.ev)
    @assert stage_axis == Base.OneTo(param.num_stages)
    variable_axis = Base.OneTo(param.num_variables)
    @assert axes(jacobian) == (output_axis, variable_axis)
    @assert axes(A) == (stage_axis, stage_axis)
    @assert axes(b) == (stage_axis,)
    @assert axes(x) == (variable_axis,)

    param(jacobian, A, b, adj.ev, x)
    return jacobian
end


function (adj::AbstractRKOCAdjoint{T})(
    param::AbstractRKParameterization{T},
    x::AbstractVector{T},
) where {T}

    # Validate array dimensions.
    stage_axis, _, output_axis = get_axes(adj.ev)
    @assert stage_axis == Base.OneTo(param.num_stages)
    variable_axis = Base.OneTo(param.num_variables)
    @assert axes(x) == (variable_axis,)

    return param(similar(Matrix{T}, output_axis, variable_axis),
        similar(Matrix{T}, stage_axis, stage_axis),
        similar(Vector{T}, stage_axis), adj.ev, x)
end


end # module AbstractRKInterface