module RungeKuttaToolKit


using MultiFloats: MultiFloat, rsqrt


include("ButcherInstructions.jl")
include("ExampleMethods.jl")
using .ButcherInstructions: LevelSequence,
    NULL_INDEX, ButcherInstruction, ButcherInstructionTable,
    rooted_trees, all_rooted_trees, butcher_density, butcher_symmetry
export LevelSequence, ButcherInstruction, ButcherInstructionTable,
    rooted_trees, all_rooted_trees, butcher_density, butcher_symmetry


####################################################### EVALUATOR DATA STRUCTURE


export RKOCEvaluator


struct RKOCEvaluator{T}
    table::ButcherInstructionTable
    Phi::Matrix{T}
    dPhi::Matrix{T}
    inv_gamma::Vector{T}
end


"""
    RKOCEvaluator{T}(
        trees::AbstractVector{LevelSequence},
        num_stages::Int;
        optimize::Bool=true,
        sort_by_depth::Bool=true,
    ) -> RKOCEvaluator{T}

Construct an `RKOCEvaluator` that encodes a given sequence of rooted trees.

# Arguments
- `trees`: input vector of rooted trees in `LevelSequence` representation.
- `num_stages`: number of stages (i.e., size of the Butcher tableau). Must be
    known at construction time to allocate internal workspace arrays.
- `optimize`: if `true`, perform common subtree elimination. This may improve
    performance in cases where `trees` is not the complete set of all rooted
    trees up to a certain order.
- `sort_by_depth`: if `true`, permute internal workspace arrays so that
    intermediate results are calculated in an order that enables parallel
    execution. This has no effect on single-threaded execution and is provided
    for forward compatibility with future parallel implementations.

Note that different permutations of `trees`, in addition to different values
of `optimize` and `sort_by_depth`, may yield slightly different results due to
the non-associative nature of floating-point arithmetic.
"""
function RKOCEvaluator{T}(
    trees::AbstractVector{LevelSequence},
    num_stages::Int;
    optimize::Bool=true,
    sort_by_depth::Bool=true,
) where {T}
    table = ButcherInstructionTable(trees;
        optimize=optimize, sort_by_depth=sort_by_depth)
    return RKOCEvaluator{T}(table,
        Matrix{T}(undef, num_stages, length(table.instructions)),
        Matrix{T}(undef, num_stages, length(table.instructions)),
        [inv(T(butcher_density(tree))) for tree in trees])
end


"""
    RKOCEvaluator{T}(order::Int, num_stages::Int) -> RKOCEvaluator{T}

Construct an `RKOCEvaluator` that encodes all rooted trees having at most
`order` vertices.

By default, rooted trees are generated in graded reverse lexicographic order
of their level sequence representation. This specific ordering maximizes the
efficiency of generating all rooted trees.
"""
@inline RKOCEvaluator{T}(order::Int, num_stages::Int) where {T} =
    RKOCEvaluator{T}(all_rooted_trees(order), num_stages;
        optimize=false, sort_by_depth=false)


################################################################################


function get_axes(ev::RKOCEvaluator{T}) where {T}
    stage_axis = axes(ev.Phi, 1)
    internal_axis = axes(ev.Phi, 2)
    output_axis = axes(ev.inv_gamma, 1)
    @assert axes(ev.table.instructions) == (internal_axis,)
    @assert axes(ev.table.selected_indices) == (output_axis,)
    @assert axes(ev.table.source_indices) == (internal_axis,)
    @assert axes(ev.table.extension_indices) == (internal_axis,)
    @assert axes(ev.table.rooted_sum_ranges) == (internal_axis,)
    @assert axes(ev.Phi) == (stage_axis, internal_axis)
    @assert axes(ev.dPhi) == (stage_axis, internal_axis)
    @assert axes(ev.inv_gamma) == (output_axis,)
    return (stage_axis, internal_axis, output_axis)
end


################################################### PHI AND RESIDUAL COMPUTATION


function compute_Phi!(
    ev::RKOCEvaluator{T},
    A::AbstractMatrix{T},
) where {T}

    # Validate array dimensions.
    stage_axis, _, _ = get_axes(ev)
    @assert axes(A) == (stage_axis, stage_axis)

    # Construct numeric constants.
    _zero = zero(T)
    _one = one(T)

    # Iterate over Butcher instructions.
    @inbounds for (k, instruction) in pairs(ev.table.instructions)
        p, q = instruction.left, instruction.right
        if p == NULL_INDEX
            # The Butcher weight vector for the trivial tree
            # (p == NULL_INDEX and q == NULL_INDEX) is the all-ones vector.
            @assert q == NULL_INDEX
            @simd ivdep for j in stage_axis
                ev.Phi[j, k] = _one
            end
        elseif q == NULL_INDEX
            # The Butcher weight vector for a tree obtained by extension
            # (p != NULL_INDEX and q == NULL_INDEX) is a matrix-vector product.
            @simd ivdep for j in stage_axis
                ev.Phi[j, k] = _zero
            end
            for i in stage_axis
                phi = ev.Phi[i, p]
                @simd ivdep for j in stage_axis
                    ev.Phi[j, k] += A[j, i] * phi
                end
            end
        else
            # The Butcher weight vector for a tree obtained by rooted sum
            # (p != NULL_INDEX and q != NULL_INDEX) is an elementwise product.
            @simd ivdep for j in stage_axis
                ev.Phi[j, k] = ev.Phi[j, p] * ev.Phi[j, q]
            end
        end
    end

    return ev
end


function compute_residuals!(
    residuals::AbstractVector{T},
    ev::RKOCEvaluator{T},
    b::AbstractVector{T},
) where {T}

    # Validate array dimensions.
    stage_axis, _, output_axis = get_axes(ev)
    @assert axes(residuals) == (output_axis,)
    @assert axes(b) == (stage_axis,)

    # Construct numeric constants.
    _zero = zero(T)

    # Compute residuals.
    @inbounds for (i, k) in pairs(ev.table.selected_indices)
        # Compute dot product without SIMD for determinism.
        lhs = _zero
        for j in stage_axis
            lhs += b[j] * ev.Phi[j, k]
        end
        # Subtract inv_gamma at the end for improved numerical stability.
        residuals[i] = lhs - ev.inv_gamma[i]
    end

    return residuals
end


"""
    (ev::RKOCEvaluator{T})(
        residuals::AbstractVector{T},
        A::AbstractMatrix{T},
        b::AbstractVector{T}
    ) -> AbstractVector{T}

Compute residuals of the Runge--Kutta order conditions
``\\{ \\mathbf{b} \\cdot \\Phi_t(A) - 1/\\gamma(t) : t \\in T \\}``
for a given Butcher tableau ``(A, \\mathbf{b})``
over a set of rooted trees ``T`` encoded by an `RKOCEvaluator`.

# Arguments
- `ev`: `RKOCEvaluator` object encoding a set of rooted trees.
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
function (ev::RKOCEvaluator{T})(
    residuals::AbstractVector{T},
    A::AbstractMatrix{T},
    b::AbstractVector{T}
) where {T}
    compute_Phi!(ev, A)
    return compute_residuals!(residuals, ev, b)
end


function residual_sum_of_squares(
    ev::RKOCEvaluator{T},
    b::AbstractVector{T},
) where {T}

    # Validate array dimensions.
    stage_axis, _, _ = get_axes(ev)
    @assert axes(b) == (stage_axis,)

    # Construct numeric constants.
    _zero = zero(T)

    # Compute sum of squared residuals.
    result = _zero
    @inbounds for (i, k) in pairs(ev.table.selected_indices)
        # Compute dot product without SIMD for determinism.
        lhs = _zero
        for j in stage_axis
            lhs += b[j] * ev.Phi[j, k]
        end
        # Subtract inv_gamma at the end for improved numerical stability.
        result += abs2(lhs - ev.inv_gamma[i])
    end

    return result
end


"""
    (ev::RKOCEvaluator{T})(
        A::AbstractMatrix{T},
        b::AbstractVector{T}
    ) -> T

Compute the sum of squared residuals of the Runge--Kutta order conditions
``\\sum_{t \\in T} (\\mathbf{b} \\cdot \\Phi_t(A) - 1/\\gamma(t))^2``
for a given Butcher tableau ``(A, \\mathbf{b})``
over a set of rooted trees ``T`` encoded by an `RKOCEvaluator`.

# Arguments
- `ev`: `RKOCEvaluator` object encoding a set of rooted trees.
- `A`: ``s \\times s`` input matrix containing the coefficients of a
    Runge--Kutta method (i.e., the upper-right block of a Butcher tableau).
- `b`: length ``s`` input vector containing the weights of a Runge--Kutta
    method (i.e., the lower-right row of a Butcher tableau).

Here, ``s`` denotes the number of stages specified when constructing `ev`.
"""
function (ev::RKOCEvaluator{T})(
    A::AbstractMatrix{T},
    b::AbstractVector{T},
) where {T}
    compute_Phi!(ev, A)
    return residual_sum_of_squares(ev, b)
end


######################################################### ADJOINT DATA STRUCTURE


struct RKOCAdjoint{T}
    ev::RKOCEvaluator{T}
end


@inline Base.adjoint(ev::RKOCEvaluator{T}) where {T} = RKOCAdjoint{T}(ev)


############################## DIRECTIONAL DERIVATIVE COMPUTATION (FORWARD-MODE)


function pushforward_dPhi!(
    ev::RKOCEvaluator{T},
    A::AbstractMatrix{T},
    dA::AbstractMatrix{T},
) where {T}

    # Validate array dimensions.
    stage_axis, _, _ = get_axes(ev)
    @assert axes(A) == (stage_axis, stage_axis)
    @assert axes(dA) == (stage_axis, stage_axis)

    # Construct numeric constants.
    _zero = zero(T)

    # Iterate over Butcher instructions.
    @inbounds for (k, instruction) in pairs(ev.table.instructions)
        p, q = instruction.left, instruction.right
        if p == NULL_INDEX
            # Trivial tree; derivative is zero.
            @assert q == NULL_INDEX
            @simd ivdep for j in stage_axis
                ev.dPhi[j, k] = _zero
            end
        elseif q == NULL_INDEX
            # Extension; apply product rule to matrix-vector product.
            @simd ivdep for j in stage_axis
                ev.dPhi[j, k] = _zero
            end
            for i in stage_axis
                phi = ev.Phi[i, p]
                dphi = ev.dPhi[i, p]
                @simd ivdep for j in stage_axis
                    ev.dPhi[j, k] += dA[j, i] * phi + A[j, i] * dphi
                end
            end
        else
            # Rooted sum; apply product rule to elementwise product.
            @simd ivdep for j in stage_axis
                ev.dPhi[j, k] = ev.dPhi[j, p] * ev.Phi[j, q] +
                                ev.Phi[j, p] * ev.dPhi[j, q]
            end
        end
    end

    return ev
end


function pushforward_dresiduals!(
    dresiduals::AbstractVector{T},
    ev::RKOCEvaluator{T},
    b::AbstractVector{T},
    db::AbstractVector{T},
) where {T}

    # Validate array dimensions.
    stage_axis, _, output_axis = get_axes(ev)
    @assert axes(dresiduals) == (output_axis,)
    @assert axes(b) == (stage_axis,)
    @assert axes(db) == (stage_axis,)

    # Construct numeric constants.
    _zero = zero(T)

    @inbounds for (i, k) in pairs(ev.table.selected_indices)
        # Compute dot product without SIMD for determinism.
        dlhs = _zero
        for j in stage_axis
            dlhs += db[j] * ev.Phi[j, k] + b[j] * ev.dPhi[j, k]
        end
        dresiduals[i] = dlhs
    end

    return dresiduals
end


"""
    (adj::RKOCAdjoint{T})(
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
over a set of rooted trees ``T`` encoded by an `RKOCEvaluator`.

# Arguments
- `adj`: `RKOCAdjoint` object obtained by applying the adjoint operator `'`
    to an `RKOCEvaluator`. In other words, this function should be called as
    `ev'(dresiduals, A, dA, b, db)` where `ev` is an `RKOCEvaluator`.
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
function (adj::RKOCAdjoint{T})(
    dresiduals::AbstractVector{T},
    A::AbstractMatrix{T},
    dA::AbstractMatrix{T},
    b::AbstractVector{T},
    db::AbstractVector{T},
) where {T}
    compute_Phi!(adj.ev, A)
    pushforward_dPhi!(adj.ev, A, dA)
    return pushforward_dresiduals!(dresiduals, adj.ev, b, db)
end


################################## PARTIAL DERIVATIVE COMPUTATION (FORWARD-MODE)


function pushforward_dPhi!(
    ev::RKOCEvaluator{T},
    A::AbstractMatrix{T},
    u::Int,
    v::Int,
) where {T}

    # Validate array dimensions.
    stage_axis, _, _ = get_axes(ev)
    @assert axes(A) == (stage_axis, stage_axis)
    @assert u in stage_axis
    @assert v in stage_axis

    # Construct numeric constants.
    _zero = zero(T)

    # Iterate over Butcher instructions.
    @inbounds for (k, instruction) in pairs(ev.table.instructions)
        p, q = instruction.left, instruction.right
        if p == NULL_INDEX
            # Trivial tree; derivative is zero.
            @assert q == NULL_INDEX
            @simd ivdep for j in stage_axis
                ev.dPhi[j, k] = _zero
            end
        elseif q == NULL_INDEX
            # Extension; apply product rule to matrix-vector product.
            @simd ivdep for j in stage_axis
                ev.dPhi[j, k] = _zero
            end
            ev.dPhi[u, k] = ev.Phi[v, p] # Additional term from product rule.
            for i in stage_axis
                dphi = ev.dPhi[i, p]
                @simd ivdep for j in stage_axis
                    ev.dPhi[j, k] += A[j, i] * dphi
                end
            end
        else
            # Rooted sum; apply product rule to elementwise product.
            @simd ivdep for j in stage_axis
                ev.dPhi[j, k] = ev.dPhi[j, p] * ev.Phi[j, q] +
                                ev.Phi[j, p] * ev.dPhi[j, q]
            end
        end
    end

    return ev
end


function pushforward_dresiduals!(
    dresiduals::AbstractVector{T},
    ev::RKOCEvaluator{T},
    b::AbstractVector{T},
) where {T}

    # Validate array dimensions.
    stage_axis, _, output_axis = get_axes(ev)
    @assert axes(dresiduals) == (output_axis,)
    @assert axes(b) == (stage_axis,)

    # Construct numeric constants.
    _zero = zero(T)

    @inbounds for (i, k) in pairs(ev.table.selected_indices)
        # Compute dot product without SIMD for determinism.
        dlhs = _zero
        for j in stage_axis
            dlhs += b[j] * ev.dPhi[j, k]
        end
        dresiduals[i] = dlhs
    end

    return dresiduals
end


"""
    (adj::RKOCAdjoint{T})(
        dresiduals::AbstractVector{T},
        A::AbstractMatrix{T},
        i::Int,
        j::Int,
        b::AbstractVector{T},
    ) -> AbstractVector{T}

Compute partial derivatives of the Runge--Kutta order conditions
``\\{ \\partial_{A_{i,j}} [ \\mathbf{b} \\cdot \\Phi_t(A) ] : t \\in T \\}``
with respect to ``A_{i,j}`` at a given Butcher tableau ``(A, \\mathbf{b})``
over a set of rooted trees ``T`` encoded by an `RKOCEvaluator`.

# Arguments
- `adj`: `RKOCAdjoint` object obtained by applying the adjoint operator `'`
    to an `RKOCEvaluator`. In other words, this function should be called as
    `ev'(dresiduals, A, i, j, b)` where `ev` is an `RKOCEvaluator`.
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
function (adj::RKOCAdjoint)(
    dresiduals::AbstractVector{T},
    A::AbstractMatrix{T},
    i::Int,
    j::Int,
    b::AbstractVector{T},
) where {T}
    compute_Phi!(adj.ev, A)
    pushforward_dPhi!(adj.ev, A, i, j)
    return pushforward_dresiduals!(dresiduals, adj.ev, b)
end


"""
    (adj::RKOCAdjoint{T})(
        dresiduals::AbstractVector{T},
        A::AbstractMatrix{T},
        i::Int,
    ) -> AbstractVector{T}

Compute partial derivatives of the Runge--Kutta order conditions
``\\{ \\partial_{b_i} [ \\mathbf{b} \\cdot \\Phi_t(A) ] : t \\in T \\}``
with respect to ``b_i`` at a given Butcher tableau ``A``
over a set of rooted trees ``T`` encoded by an `RKOCEvaluator`.
(The result is independent of the value of ``\\mathbf{b}``.)

# Arguments
- `adj`: `RKOCAdjoint` object obtained by applying the adjoint operator `'`
    to an `RKOCEvaluator`. In other words, this function should be called as
    `ev'(dresiduals, A, i)` where `ev` is an `RKOCEvaluator`.
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
function (adj::RKOCAdjoint)(
    dresiduals::AbstractVector{T},
    A::AbstractMatrix{T},
    i::Int,
) where {T}

    # Validate array dimensions.
    stage_axis, _, _ = get_axes(adj.ev)
    @assert i in stage_axis

    compute_Phi!(adj.ev, A)

    # Extract entries of Phi.
    @inbounds for (j, k) in pairs(adj.ev.table.selected_indices)
        dresiduals[j] = adj.ev.Phi[i, k]
    end

    return dresiduals
end


############################################ GRADIENT COMPUTATION (REVERSE-MODE)


function pullback_dPhi_from_b!(
    ev::RKOCEvaluator{T},
    b::AbstractVector{T},
) where {T}

    # Validate array dimensions.
    stage_axis, _, _ = get_axes(ev)
    @assert axes(b) == (stage_axis,)

    # Construct numeric constants.
    _zero = zero(T)

    @inbounds for (k, i) in Iterators.reverse(pairs(ev.table.source_indices))
        if i == NULL_INDEX
            @simd ivdep for j in stage_axis
                ev.dPhi[j, k] = _zero
            end
        else
            # Compute dot product without SIMD for determinism.
            lhs = _zero
            for j in stage_axis
                lhs += b[j] * ev.Phi[j, k]
            end
            # Subtract inv_gamma at the end for improved numerical stability.
            residual = lhs - ev.inv_gamma[i]
            # Double residual. (Addition is faster than multiplication by two.)
            twice_residual = residual + residual
            @simd ivdep for j in stage_axis
                ev.dPhi[j, k] = twice_residual * b[j]
            end
        end
    end

    return ev
end


function pullback_dPhi_from_A!(
    ev::RKOCEvaluator{T},
    A::AbstractMatrix{T},
) where {T}

    # Validate array dimensions.
    stage_axis, internal_axis, _ = get_axes(ev)
    @assert axes(A) == (stage_axis, stage_axis)

    # Iterate over intermediate trees in reverse order.
    @inbounds for k in Iterators.reverse(internal_axis)
        c = ev.table.extension_indices[k]
        if c != NULL_INDEX
            # Perform adjoint matrix-vector multiplication.
            for i in stage_axis
                dphi = ev.dPhi[i, c]
                @simd ivdep for j in stage_axis
                    ev.dPhi[j, k] += A[i, j] * dphi
                end
            end
        end
        for i in ev.table.rooted_sum_ranges[k]
            (p, q) = ev.table.rooted_sum_indices[i]
            @simd ivdep for j in stage_axis
                ev.dPhi[j, k] += ev.Phi[j, p] * ev.dPhi[j, q]
            end
        end
    end

    return ev
end


function pullback_dA!(
    dA::AbstractMatrix{T},
    ev::RKOCEvaluator{T},
) where {T}

    # Validate array dimensions.
    stage_axis, _, _ = get_axes(ev)
    @assert axes(dA) == (stage_axis, stage_axis)

    # Construct numeric constants.
    _zero = zero(T)

    # Initialize dA to zero.
    @inbounds for j in stage_axis
        @simd ivdep for i in stage_axis
            dA[i, j] = _zero
        end
    end

    # Iterate over intermediate trees obtained by extension.
    @inbounds for (k, c) in pairs(ev.table.extension_indices)
        if c != NULL_INDEX
            for t in stage_axis
                phi = ev.Phi[t, k]
                @simd ivdep for s in stage_axis
                    dA[s, t] += phi * ev.dPhi[s, c]
                end
            end
        end
    end

    return dA
end


function pullback_db!(
    db::AbstractVector{T},
    ev::RKOCEvaluator{T},
    b::AbstractVector{T},
) where {T}

    # Validate array dimensions.
    stage_axis, _, _ = get_axes(ev)
    @assert axes(db) == (stage_axis,)
    @assert axes(b) == (stage_axis,)

    # Construct numeric constants.
    _zero = zero(T)

    @inbounds begin

        # Initialize db to zero.
        @simd ivdep for i in stage_axis
            db[i] = _zero
        end

        for (i, k) in pairs(ev.table.selected_indices)
            # Compute dot product without SIMD for determinism.
            lhs = _zero
            for j in stage_axis
                lhs += b[j] * ev.Phi[j, k]
            end
            # Subtract inv_gamma at the end for improved numerical stability.
            residual = lhs - ev.inv_gamma[i]
            @simd ivdep for j in stage_axis
                db[j] += residual * ev.Phi[j, k]
            end
        end

        # Double db. (Addition is faster than multiplication by two.)
        @simd ivdep for i in stage_axis
            db[i] += db[i]
        end

    end

    return db
end


"""
    (adj::RKOCAdjoint{T})(
        dA::AbstractMatrix{T},
        db::AbstractVector{T},
        A::AbstractMatrix{T},
        b::AbstractVector{T},
    ) -> Tuple{AbstractMatrix{T}, AbstractVector{T}}

Compute the gradient of the sum of squared residuals
of the Runge--Kutta order conditions ``\\nabla_{A, \\mathbf{b}}
\\sum_{t \\in T} (\\mathbf{b} \\cdot \\Phi_t(A) - 1/\\gamma(t))^2``
at a given Butcher tableau ``(A, \\mathbf{b})``
over a set of rooted trees ``T`` encoded by an `RKOCEvaluator`.

# Arguments
- `adj`: `RKOCAdjoint` object obtained by applying the adjoint operator `'`
    to an `RKOCEvaluator`. In other words, this function should be called as
    `ev'(dA, db, A, b)` where `ev` is an `RKOCEvaluator`.
- `dA`: ``s \\times s`` output matrix containing the gradient of the sum of
    squared residuals with respect to ``A``.
- `db`: length ``s`` output vector containing the gradient of the sum of
    squared residuals with respect to ``\\mathbf{b}``.
- `A`: ``s \\times s`` input matrix containing the coefficients of a
    Runge--Kutta method (i.e., the upper-right block of a Butcher tableau).
- `b`: length ``s`` input vector containing the weights of a Runge--Kutta
    method (i.e., the lower-right row of a Butcher tableau).

Here, ``s`` denotes the number of stages specified when constructing `ev`.
"""
function (adj::RKOCAdjoint{T})(
    dA::AbstractMatrix{T},
    db::AbstractVector{T},
    A::AbstractMatrix{T},
    b::AbstractVector{T},
) where {T}
    compute_Phi!(adj.ev, A)
    pullback_dPhi_from_b!(adj.ev, b)
    pullback_dPhi_from_A!(adj.ev, A)
    pullback_dA!(dA, adj.ev)
    pullback_db!(db, adj.ev, b)
    return (dA, db)
end


########################################################### LEAST-SQUARES SOLVER


@inline inv_sqrt(x::Float16) = rsqrt(x)
@inline inv_sqrt(x::Float32) = rsqrt(x)
@inline inv_sqrt(x::Float64) = rsqrt(x)
@inline inv_sqrt(x::MultiFloat{T,N}) where {T,N} = rsqrt(x)
@inline inv_sqrt(x::T) where {T} = inv(sqrt(x))


################################################### RESHAPING COEFFICIENT ARRAYS


function reshape_explicit!(
    A::AbstractMatrix{T},
    x::AbstractVector{T},
) where {T}

    # Validate array dimensions.
    s = size(A, 1)
    @assert s == size(A, 2)
    @assert ((s * (s - 1)) >> 1,) == size(x)
    Base.require_one_based_indexing(A, x)

    # Construct numeric constants.
    _zero = zero(T)

    # Iterate over the strict lower-triangular part of A.
    offset = 0
    for i = 1:s
        @simd ivdep for j = 1:i-1
            @inbounds A[i, j] = x[offset+j]
        end
        offset += i - 1
        @simd ivdep for j = i:s
            @inbounds A[i, j] = _zero
        end
    end

    return A
end


function reshape_explicit!(
    x::AbstractVector{T},
    A::AbstractMatrix{T},
) where {T}

    # Validate array dimensions.
    s = size(A, 1)
    @assert s == size(A, 2)
    @assert ((s * (s - 1)) >> 1,) == size(x)
    Base.require_one_based_indexing(x, A)

    # Iterate over the strict lower-triangular part of A.
    offset = 0
    for i = 2:s
        @simd ivdep for j = 1:i-1
            @inbounds x[offset+j] = A[i, j]
        end
        offset += i - 1
    end
    return x
end


function reshape_explicit!(
    A::AbstractMatrix{T},
    b::AbstractVector{T},
    x::AbstractVector{T},
) where {T}

    # Validate array dimensions.
    s = length(b)
    @assert (s, s) == size(A)
    @assert ((s * (s + 1)) >> 1,) == size(x)
    Base.require_one_based_indexing(A, b, x)

    # Construct numeric constants.
    _zero = zero(T)

    # Iterate over the strict lower-triangular part of A.
    offset = 0
    for i = 1:s
        @simd ivdep for j = 1:i-1
            @inbounds A[i, j] = x[offset+j]
        end
        offset += i - 1
        @simd ivdep for j = i:s
            @inbounds A[i, j] = _zero
        end
    end

    # Iterate over b.
    @simd ivdep for i = 1:s
        @inbounds b[i] = x[offset+i]
    end
    return (A, b)
end


function reshape_explicit!(
    x::AbstractVector{T},
    A::AbstractMatrix{T},
    b::AbstractVector{T},
) where {T}

    # Validate array dimensions.
    s = length(b)
    @assert (s, s) == size(A)
    @assert ((s * (s + 1)) >> 1,) == size(x)
    Base.require_one_based_indexing(x, A, b)

    # Iterate over the strict lower-triangular part of A.
    offset = 0
    for i = 2:s
        @simd ivdep for j = 1:i-1
            @inbounds x[offset+j] = A[i, j]
        end
        offset += i - 1
    end

    # Iterate over b.
    @simd ivdep for i = 1:s
        @inbounds x[offset+i] = b[i]
    end
    return x
end


function reshape_diagonally_implicit!(
    A::AbstractMatrix{T},
    x::AbstractVector{T},
) where {T}

    # Validate array dimensions.
    s = size(A, 1)
    @assert s == size(A, 2)
    @assert ((s * (s + 1)) >> 1,) == size(x)
    Base.require_one_based_indexing(A, x)

    # Construct numeric constants.
    _zero = zero(T)

    # Iterate over the lower-triangular part of A.
    offset = 0
    for i = 1:s
        @simd ivdep for j = 1:i
            @inbounds A[i, j] = x[offset+j]
        end
        offset += i
        @simd ivdep for j = i+1:s
            @inbounds A[i, j] = _zero
        end
    end
    return A
end


function reshape_diagonally_implicit!(
    x::AbstractVector{T},
    A::AbstractMatrix{T},
) where {T}

    # Validate array dimensions.
    s = size(A, 1)
    @assert s == size(A, 2)
    @assert ((s * (s + 1)) >> 1,) == size(x)
    Base.require_one_based_indexing(x, A)

    # Iterate over the lower-triangular part of A.
    offset = 0
    for i = 1:s
        @simd ivdep for j = 1:i
            @inbounds x[offset+j] = A[i, j]
        end
        offset += i
    end
    return x
end


function reshape_diagonally_implicit!(
    A::AbstractMatrix{T},
    b::AbstractVector{T},
    x::AbstractVector{T},
) where {T}

    # Validate array dimensions.
    s = length(b)
    @assert (s, s) == size(A)
    @assert ((s * (s + 3)) >> 1,) == size(x)
    Base.require_one_based_indexing(A, b, x)

    # Construct numeric constants.
    _zero = zero(T)

    # Iterate over the lower-triangular part of A.
    offset = 0
    for i = 1:s
        @simd ivdep for j = 1:i
            @inbounds A[i, j] = x[offset+j]
        end
        offset += i
        @simd ivdep for j = i+1:s
            @inbounds A[i, j] = _zero
        end
    end

    # Iterate over b.
    @simd ivdep for i = 1:s
        @inbounds b[i] = x[offset+i]
    end
    return (A, b)
end


function reshape_diagonally_implicit!(
    x::AbstractVector{T},
    A::AbstractMatrix{T},
    b::AbstractVector{T},
) where {T}

    # Validate array dimensions.
    s = length(b)
    @assert (s, s) == size(A)
    @assert ((s * (s + 3)) >> 1,) == size(x)
    Base.require_one_based_indexing(x, A, b)

    # Iterate over the lower-triangular part of A.
    offset = 0
    for i = 1:s
        @simd ivdep for j = 1:i
            @inbounds x[offset+j] = A[i, j]
        end
        offset += i
    end

    # Iterate over b.
    @simd ivdep for i = 1:s
        @inbounds x[offset+i] = b[i]
    end
    return x
end


function reshape_implicit!(
    A::AbstractMatrix{T},
    x::AbstractVector{T},
) where {T}

    # Validate array dimensions.
    s = size(A, 1)
    @assert s == size(A, 2)
    @assert (s * s,) == size(x)
    Base.require_one_based_indexing(A, x)

    # Iterate over A.
    offset = 0
    for i = 1:s
        @simd ivdep for j = 1:s
            @inbounds A[i, j] = x[offset+j]
        end
        offset += s
    end
    return A
end


function reshape_implicit!(
    x::AbstractVector{T},
    A::AbstractMatrix{T},
) where {T}

    # Validate array dimensions.
    s = size(A, 1)
    @assert s == size(A, 2)
    @assert (s * s,) == size(x)
    Base.require_one_based_indexing(x, A)

    # Iterate over A.
    offset = 0
    for i = 1:s
        @simd ivdep for j = 1:s
            @inbounds x[offset+j] = A[i, j]
        end
        offset += s
    end
    return x
end


function reshape_implicit!(
    A::AbstractMatrix{T},
    b::AbstractVector{T},
    x::AbstractVector{T},
) where {T}

    # Validate array dimensions.
    s = size(A, 1)
    @assert s == size(A, 2)
    @assert (s * (s + 1),) == size(x)
    Base.require_one_based_indexing(A, b, x)

    # Iterate over A.
    offset = 0
    for i = 1:s
        @simd ivdep for j = 1:s
            @inbounds A[i, j] = x[offset+j]
        end
        offset += s
    end

    # Iterate over b.
    @simd ivdep for i = 1:s
        @inbounds b[i] = x[offset+i]
    end
    return (A, b)
end


function reshape_implicit!(
    x::AbstractVector{T},
    A::AbstractMatrix{T},
    b::AbstractVector{T},
) where {T}

    # Validate array dimensions.
    s = size(A, 1)
    @assert s == size(A, 2)
    @assert (s * (s + 1),) == size(x)
    Base.require_one_based_indexing(x, A, b)

    # Iterate over A.
    offset = 0
    for i = 1:s
        @simd ivdep for j = 1:s
            @inbounds x[offset+j] = A[i, j]
        end
        offset += s
    end

    # Iterate over b.
    copy!
    @simd ivdep for i = 1:s
        @inbounds x[offset+i] = b[i]
    end
    return x
end


############################################################## FUNCTOR INTERFACE


# function populate_Q!(
#     Q::AbstractMatrix{T}, Phi::AbstractMatrix{T},
#     selected_indices::AbstractVector{Int}
# ) where {T}
#     t, s = size(Q)
#     @assert s == size(Phi, 1)
#     @assert (t,) == size(selected_indices)
#     for (i, k) in pairs(selected_indices)
#         @simd ivdep for j = 1:s
#             @inbounds Q[i, j] = Phi[j, k]
#         end
#     end
# end


# function (ev::RKOCEvaluatorAE{T})(x::Vector{T}) where {T}
#     reshape_explicit!(ev.A, x)
#     compute_Phi!(ev.Phi, ev.A, ev.table.instructions)
#     populate_Q!(ev.Q, ev.Phi, ev.table.selected_indices)
#     gram_schmidt_qr!(ev.Q)
#     compute_residuals!(ev.residuals, ev.Q, ev.inv_gamma)
#     return residual_norm_squared(ev.residuals)
# end


# function (ev::RKOCEvaluatorAD{T})(x::Vector{T}) where {T}
#     reshape_diagonally_implicit!(ev.A, x)
#     compute_Phi!(ev.Phi, ev.A, ev.table.instructions)
#     populate_Q!(ev.Q, ev.Phi, ev.table.selected_indices)
#     gram_schmidt_qr!(ev.Q)
#     compute_residuals!(ev.residuals, ev.Q, ev.inv_gamma)
#     return residual_norm_squared(ev.residuals)
# end


# function (ev::RKOCEvaluatorAI{T})(x::Vector{T}) where {T}
#     reshape_implicit!(ev.A, x)
#     compute_Phi!(ev.Phi, ev.A, ev.table.instructions)
#     populate_Q!(ev.Q, ev.Phi, ev.table.selected_indices)
#     gram_schmidt_qr!(ev.Q)
#     compute_residuals!(ev.residuals, ev.Q, ev.inv_gamma)
#     return residual_norm_squared(ev.residuals)
# end


# function (ev::RKOCResidualEvaluatorAE{T})(
#     residuals::Vector{T}, x::Vector{T}
# ) where {T}
#     reshape_explicit!(ev.A, x)
#     compute_Phi!(ev.Phi, ev.A, ev.table.instructions)
#     populate_Q!(ev.Q, ev.Phi, ev.table.selected_indices)
#     gram_schmidt_qr!(ev.Q)
#     compute_residuals!(residuals, ev.Q, ev.inv_gamma)
#     return residuals
# end


# function (ev::RKOCResidualEvaluatorAD{T})(
#     residuals::Vector{T}, x::Vector{T}
# ) where {T}
#     reshape_diagonally_implicit!(ev.A, x)
#     compute_Phi!(ev.Phi, ev.A, ev.table.instructions)
#     populate_Q!(ev.Q, ev.Phi, ev.table.selected_indices)
#     gram_schmidt_qr!(ev.Q)
#     compute_residuals!(residuals, ev.Q, ev.inv_gamma)
#     return residuals
# end


# function (ev::RKOCResidualEvaluatorAI{T})(
#     residuals::Vector{T}, x::Vector{T}
# ) where {T}
#     reshape_implicit!(ev.A, x)
#     compute_Phi!(ev.Phi, ev.A, ev.table.instructions)
#     populate_Q!(ev.Q, ev.Phi, ev.table.selected_indices)
#     gram_schmidt_qr!(ev.Q)
#     compute_residuals!(residuals, ev.Q, ev.inv_gamma)
#     return residuals
# end


# function (adj::RKOCEvaluatorAEAdjoint{T})(g::Vector{T}, x::Vector{T}) where {T}
#     reshape_explicit!(adj.ev.A, x)
#     compute_Phi!(adj.ev.Phi, adj.ev.A, adj.ev.table.instructions)
#     populate_Q!(adj.ev.Q, adj.ev.Phi, adj.ev.table.selected_indices)
#     gram_schmidt_qr!(adj.ev.Q, adj.ev.R)
#     compute_residuals_and_b!(adj.ev.residuals, adj.ev.b,
#         adj.ev.Q, adj.ev.inv_gamma)
#     solve_upper_triangular!(adj.ev.b, adj.ev.R)
#     pullback_dPhi_from_residual!(adj.ev.dPhi,
#         adj.ev.b, adj.ev.residuals, adj.ev.table.source_indices)
#     pullback_dPhi!(adj.ev.dPhi,
#         adj.ev.A, adj.ev.Phi, adj.ev.table.extension_indices,
#         adj.ev.table.rooted_sum_ranges, adj.ev.table.rooted_sum_indices)
#     pullback_dA!(adj.ev.dA,
#         adj.ev.Phi, adj.ev.dPhi, adj.ev.table.extension_indices)
#     reshape_explicit!(g, adj.ev.dA)
#     return g
# end


# function (adj::RKOCEvaluatorADAdjoint{T})(g::Vector{T}, x::Vector{T}) where {T}
#     reshape_diagonally_implicit!(adj.ev.A, x)
#     compute_Phi!(adj.ev.Phi, adj.ev.A, adj.ev.table.instructions)
#     populate_Q!(adj.ev.Q, adj.ev.Phi, adj.ev.table.selected_indices)
#     gram_schmidt_qr!(adj.ev.Q, adj.ev.R)
#     compute_residuals_and_b!(adj.ev.residuals, adj.ev.b,
#         adj.ev.Q, adj.ev.inv_gamma)
#     solve_upper_triangular!(adj.ev.b, adj.ev.R)
#     pullback_dPhi_from_residual!(adj.ev.dPhi,
#         adj.ev.b, adj.ev.residuals, adj.ev.table.source_indices)
#     pullback_dPhi!(adj.ev.dPhi,
#         adj.ev.A, adj.ev.Phi, adj.ev.table.extension_indices,
#         adj.ev.table.rooted_sum_ranges, adj.ev.table.rooted_sum_indices)
#     pullback_dA!(adj.ev.dA,
#         adj.ev.Phi, adj.ev.dPhi, adj.ev.table.extension_indices)
#     reshape_diagonally_implicit!(g, adj.ev.dA)
#     return g
# end


# function (adj::RKOCEvaluatorAIAdjoint{T})(g::Vector{T}, x::Vector{T}) where {T}
#     reshape_implicit!(adj.ev.A, x)
#     compute_Phi!(adj.ev.Phi, adj.ev.A, adj.ev.table.instructions)
#     populate_Q!(adj.ev.Q, adj.ev.Phi, adj.ev.table.selected_indices)
#     gram_schmidt_qr!(adj.ev.Q, adj.ev.R)
#     compute_residuals_and_b!(adj.ev.residuals, adj.ev.b,
#         adj.ev.Q, adj.ev.inv_gamma)
#     solve_upper_triangular!(adj.ev.b, adj.ev.R)
#     pullback_dPhi_from_residual!(adj.ev.dPhi,
#         adj.ev.b, adj.ev.residuals, adj.ev.table.source_indices)
#     pullback_dPhi!(adj.ev.dPhi,
#         adj.ev.A, adj.ev.Phi, adj.ev.table.extension_indices,
#         adj.ev.table.rooted_sum_ranges, adj.ev.table.rooted_sum_indices)
#     pullback_dA!(adj.ev.dA,
#         adj.ev.Phi, adj.ev.dPhi, adj.ev.table.extension_indices)
#     reshape_implicit!(g, adj.ev.dA)
#     return g
# end


# function (adj::RKOCResidualEvaluatorAEAdjoint{T})(
#     jacobian::Matrix{T}, x::Vector{T}
# ) where {T}
#     @assert length(adj.ev.table.selected_indices) == size(jacobian, 1)
#     @assert length(x) == size(jacobian, 2)
#     reshape_explicit!(adj.ev.A, x)
#     compute_Phi!(adj.ev.Phi, adj.ev.A, adj.ev.table.instructions)
#     populate_Q!(adj.ev.Q, adj.ev.Phi, adj.ev.table.selected_indices)
#     gram_schmidt_qr!(adj.ev.Q, adj.ev.R)
#     compute_residuals_and_b!(adj.ev.residuals, adj.ev.b,
#         adj.ev.Q, adj.ev.inv_gamma)
#     solve_upper_triangular!(adj.ev.b, adj.ev.R)
#     s = length(adj.ev.b)
#     k = 1
#     for i = 2:s
#         for j = 1:i-1
#             pushforward_dPhi!(adj.ev.dPhi,
#                 adj.ev.Phi, adj.ev.A, i, j, adj.ev.table.instructions)
#             column = view(jacobian, :, k)
#             pushforward_db!(adj.ev.db, column,
#                 adj.ev.residuals, adj.ev.dPhi, adj.ev.b, adj.ev.Q, adj.ev.R,
#                 adj.ev.table.selected_indices)
#             pushforward_dresiduals!(column,
#                 adj.ev.db, adj.ev.b, adj.ev.dPhi, adj.ev.Phi,
#                 adj.ev.table.selected_indices)
#             k += 1
#         end
#     end
#     return jacobian
# end


# function (adj::RKOCResidualEvaluatorADAdjoint{T})(
#     jacobian::Matrix{T}, x::Vector{T}
# ) where {T}
#     @assert length(adj.ev.table.selected_indices) == size(jacobian, 1)
#     @assert length(x) == size(jacobian, 2)
#     reshape_diagonally_implicit!(adj.ev.A, x)
#     compute_Phi!(adj.ev.Phi, adj.ev.A, adj.ev.table.instructions)
#     populate_Q!(adj.ev.Q, adj.ev.Phi, adj.ev.table.selected_indices)
#     gram_schmidt_qr!(adj.ev.Q, adj.ev.R)
#     compute_residuals_and_b!(adj.ev.residuals, adj.ev.b,
#         adj.ev.Q, adj.ev.inv_gamma)
#     solve_upper_triangular!(adj.ev.b, adj.ev.R)
#     s = length(adj.ev.b)
#     k = 1
#     for i = 1:s
#         for j = 1:i
#             pushforward_dPhi!(adj.ev.dPhi,
#                 adj.ev.Phi, adj.ev.A, i, j, adj.ev.table.instructions)
#             column = view(jacobian, :, k)
#             pushforward_db!(adj.ev.db, column,
#                 adj.ev.residuals, adj.ev.dPhi, adj.ev.b, adj.ev.Q, adj.ev.R,
#                 adj.ev.table.selected_indices)
#             pushforward_dresiduals!(column,
#                 adj.ev.db, adj.ev.b, adj.ev.dPhi, adj.ev.Phi,
#                 adj.ev.table.selected_indices)
#             k += 1
#         end
#     end
#     return jacobian
# end


# function (adj::RKOCResidualEvaluatorAIAdjoint{T})(
#     jacobian::Matrix{T}, x::Vector{T}
# ) where {T}
#     @assert length(adj.ev.table.selected_indices) == size(jacobian, 1)
#     @assert length(x) == size(jacobian, 2)
#     reshape_implicit!(adj.ev.A, x)
#     compute_Phi!(adj.ev.Phi, adj.ev.A, adj.ev.table.instructions)
#     populate_Q!(adj.ev.Q, adj.ev.Phi, adj.ev.table.selected_indices)
#     gram_schmidt_qr!(adj.ev.Q, adj.ev.R)
#     compute_residuals_and_b!(adj.ev.residuals, adj.ev.b,
#         adj.ev.Q, adj.ev.inv_gamma)
#     solve_upper_triangular!(adj.ev.b, adj.ev.R)
#     s = length(adj.ev.b)
#     k = 1
#     for i = 1:s
#         for j = 1:s
#             pushforward_dPhi!(adj.ev.dPhi,
#                 adj.ev.Phi, adj.ev.A, i, j, adj.ev.table.instructions)
#             column = view(jacobian, :, k)
#             pushforward_db!(adj.ev.db, column,
#                 adj.ev.residuals, adj.ev.dPhi, adj.ev.b, adj.ev.Q, adj.ev.R,
#                 adj.ev.table.selected_indices)
#             pushforward_dresiduals!(column,
#                 adj.ev.db, adj.ev.b, adj.ev.dPhi, adj.ev.Phi,
#                 adj.ev.table.selected_indices)
#             k += 1
#         end
#     end
#     return jacobian
# end


# function (adj::RKOCResidualEvaluatorBEAdjoint{T})(
#     jacobian::Matrix{T}, x::Vector{T}
# ) where {T}
#     @assert length(adj.ev.table.selected_indices) == size(jacobian, 1)
#     @assert length(x) == size(jacobian, 2)
#     reshape_explicit!(adj.ev.A, adj.ev.b, x)
#     compute_Phi!(adj.ev.Phi, adj.ev.A, adj.ev.table.instructions)
#     s = length(adj.ev.b)
#     k = 1
#     for i = 2:s
#         for j = 1:i-1
#             pushforward_dPhi!(adj.ev.dPhi,
#                 adj.ev.Phi, adj.ev.A, i, j, adj.ev.table.instructions)
#             pushforward_dresiduals!(view(jacobian, :, k),
#                 adj.ev.b, adj.ev.dPhi, adj.ev.table.selected_indices)
#             k += 1
#         end
#     end
#     return jacobian
# end


# function (adj::RKOCResidualEvaluatorBDAdjoint{T})(
#     jacobian::Matrix{T}, x::Vector{T}
# ) where {T}
#     @assert length(adj.ev.table.selected_indices) == size(jacobian, 1)
#     @assert length(x) == size(jacobian, 2)
#     reshape_diagonally_implicit!(adj.ev.A, adj.ev.b, x)
#     compute_Phi!(adj.ev.Phi, adj.ev.A, adj.ev.table.instructions)
#     s = length(adj.ev.b)
#     k = 1
#     for i = 1:s
#         for j = 1:i
#             pushforward_dPhi!(adj.ev.dPhi,
#                 adj.ev.Phi, adj.ev.A, i, j, adj.ev.table.instructions)
#             pushforward_dresiduals!(view(jacobian, :, k),
#                 adj.ev.b, adj.ev.dPhi, adj.ev.table.selected_indices)
#             k += 1
#         end
#     end
#     for j = 1:s
#         for (i, m) in pairs(adj.ev.table.selected_indices)
#             @inbounds jacobian[i, k] = adj.ev.Phi[j, m]
#         end
#         k += 1
#     end
#     return jacobian
# end


# function (adj::RKOCResidualEvaluatorBIAdjoint{T})(
#     jacobian::Matrix{T}, x::Vector{T}
# ) where {T}
#     @assert length(adj.ev.table.selected_indices) == size(jacobian, 1)
#     @assert length(x) == size(jacobian, 2)
#     reshape_implicit!(adj.ev.A, adj.ev.b, x)
#     compute_Phi!(adj.ev.Phi, adj.ev.A, adj.ev.table.instructions)
#     s = length(adj.ev.b)
#     k = 1
#     for i = 1:s
#         for j = 1:s
#             pushforward_dPhi!(adj.ev.dPhi,
#                 adj.ev.Phi, adj.ev.A, i, j, adj.ev.table.instructions)
#             pushforward_dresiduals!(view(jacobian, :, k),
#                 adj.ev.b, adj.ev.dPhi, adj.ev.table.selected_indices)
#             k += 1
#         end
#     end
#     for j = 1:s
#         for (i, m) in pairs(adj.ev.table.selected_indices)
#             @inbounds jacobian[i, k] = adj.ev.Phi[j, m]
#         end
#         k += 1
#     end
#     return jacobian
# end


end # module RungeKuttaToolKit
