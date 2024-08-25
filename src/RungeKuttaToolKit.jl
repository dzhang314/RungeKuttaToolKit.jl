module RungeKuttaToolKit


using MultiFloats: MultiFloat, rsqrt


include("ButcherInstructions.jl")
using .ButcherInstructions: LevelSequence,
    NULL_INDEX, ButcherInstruction, ButcherInstructionTable,
    rooted_trees, all_rooted_trees, butcher_density, butcher_symmetry
export LevelSequence, ButcherInstruction, ButcherInstructionTable,
    rooted_trees, all_rooted_trees, butcher_density, butcher_symmetry


###################################################### EVALUATOR DATA STRUCTURES


export RKOCEvaluator, RKOCEvaluatorQR


struct RKOCEvaluator{T}
    table::ButcherInstructionTable
    Phi::Matrix{T}
    dPhi::Matrix{T}
    inv_gamma::Vector{T}
end


function RKOCEvaluator{T}(
    trees::AbstractVector{LevelSequence}, s::Int
) where {T}
    table = ButcherInstructionTable(trees)
    return RKOCEvaluator{T}(table,
        Matrix{T}(undef, s, length(table.instructions)),
        Matrix{T}(undef, s, length(table.instructions)),
        [inv(T(butcher_density(tree))) for tree in trees])
end


@inline RKOCEvaluator{T}(p::Int, s::Int) where {T} =
    RKOCEvaluator{T}(all_rooted_trees(p), s)


struct RKOCEvaluatorQR{T}
    table::ButcherInstructionTable
    Phi::Matrix{T}
    dPhi::Matrix{T}
    Q::Matrix{T}
    R::Matrix{T}
    inv_gamma::Vector{T}
end


function RKOCEvaluatorQR{T}(trees::Vector{LevelSequence}, s::Int) where {T}
    table = ButcherInstructionTable(trees)
    return RKOCEvaluatorQR{T}(table,
        Matrix{T}(undef, s, length(table.instructions)),
        Matrix{T}(undef, s, length(table.instructions)),
        Matrix{T}(undef, length(trees), s),
        Matrix{T}(undef, s, s),
        [inv(T(butcher_density(tree))) for tree in trees])
end


@inline RKOCEvaluatorQR{T}(p::Int, s::Int) where {T} =
    RKOCEvaluatorQR{T}(all_rooted_trees(p), s)


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


"""
    RungeKuttaToolKit.compute_Phi!(
        Phi::AbstractMatrix{T},
        A::AbstractMatrix{T},
        instructions::AbstractVector{ButcherInstruction},
    ) -> Phi

Compute Butcher weight vectors ``\\{ \\Phi_t(A) : t \\in T \\}`` of a given
matrix ``A`` over a set of rooted trees ``T`` represented by `instructions`.

# Arguments
- `Phi`: ``s \\times N`` output matrix. Each Butcher weight vector
    ``\\Phi_t(A)`` is written to `Phi[:, i]`, where ``t`` is the rooted tree
    represented by `instructions[i]`.
- `A`: ``s \\times s`` input matrix containing Runge--Kutta coefficients.
- `instructions`: length ``N`` input vector of `ButcherInstruction` objects
    encoding a set of rooted trees.

Here, ``s`` is the number of stages in the Runge--Kutta method represented by
`A`, and ``N`` is the number of rooted trees in ``T``.
"""
function compute_Phi!(
    Phi::AbstractMatrix{T},
    A::AbstractMatrix{T},
    instructions::AbstractVector{ButcherInstruction},
) where {T}

    # Validate array dimensions.
    stage_indices = axes(Phi, 1)
    internal_indices = axes(Phi, 2)
    @assert axes(Phi) == (stage_indices, internal_indices)
    @assert axes(A) == (stage_indices, stage_indices)
    @assert axes(instructions) == (internal_indices,)

    # Construct numeric constants.
    _zero = zero(T)
    _one = one(T)

    # Iterate over Butcher instructions.
    @inbounds for (k, instruction) in pairs(instructions)
        p, q = instruction.left, instruction.right
        if p == NULL_INDEX
            # The Butcher weight vector for the trivial tree
            # (p == NULL_INDEX and q == NULL_INDEX) is the all-ones vector.
            @assert q == NULL_INDEX
            @simd ivdep for j in stage_indices
                Phi[j, k] = _one
            end
        elseif q == NULL_INDEX
            # The Butcher weight vector for a tree obtained by extension
            # (p != NULL_INDEX and q == NULL_INDEX) is a matrix-vector product.
            @simd ivdep for j in stage_indices
                Phi[j, k] = _zero
            end
            for i in stage_indices
                temp = Phi[i, p]
                @simd ivdep for j in stage_indices
                    Phi[j, k] += A[j, i] * temp
                end
            end
        else
            # The Butcher weight vector for a tree obtained by rooted sum
            # (p != NULL_INDEX and q != NULL_INDEX) is an elementwise product.
            @simd ivdep for j in stage_indices
                Phi[j, k] = Phi[j, p] * Phi[j, q]
            end
        end
    end

    return Phi
end


"""
    RungeKuttaToolKit.compute_residuals!(
        residuals::AbstractVector{T},
        b::AbstractVector{T},
        Phi::AbstractMatrix{T},
        inv_gamma::AbstractVector{T},
        selected_indices::AbstractVector{Int},
    ) where {T}

Compute residuals ``\\{ \\mathbf{b} \\cdot \\Phi_t(A) - 1/\\gamma(t) :
t \\in T \\}`` of Runge--Kutta order conditions over a set of rooted trees
``T`` using precomputed Butcher weight vectors ``\\Phi_t(A)`` and inverse
density values ``1/\\gamma(t)``.

# Arguments
- `residuals`: length ``N_{\\text{output}}`` output vector. Each residual
    ``\\mathbf{b} \\cdot \\Phi_t(A) - 1/\\gamma(t)`` is written to
    `residuals[i]`, where ``t`` is the rooted tree whose Butcher weight vector
    is precomputed in `Phi[:, selected_indices[i]]`.
- `b`: length ``s`` input vector containing Runge--Kutta weights.
- `Phi`: ``s \\times N_{\\text{internal}}`` input matrix containing
    precomputed Butcher weight vectors ``\\Phi_t(A)``. `Phi` may contain
    additional Butcher weight vectors corresponding to rooted trees not in
    ``T``. This is necessary when ``T`` contains a tree but not its subtrees.
- `inv_gamma`: length ``N_{\\text{output}}`` input vector containing
    precomputed Butcher density values.
- `selected_indices`: length ``N_{\\text{output}}`` input vector of indices
    in the range ``1:N_{\\text{internal}}`` specifying Butcher weight
    vectors in `Phi` corresponding to rooted trees in ``T``.

Here, ``s`` is the number of stages in the Runge--Kutta method represented by
`b`, ``N_{\\text{output}}`` is the number of rooted trees in ``T``, and
``N_{\\text{internal}}`` is the number of Butcher weight vectors in `Phi`.
"""
function compute_residuals!(
    residuals::AbstractVector{T},
    b::AbstractVector{T},
    Phi::AbstractMatrix{T},
    inv_gamma::AbstractVector{T},
    selected_indices::AbstractVector{Int},
) where {T}

    # Validate array dimensions.
    output_indices = axes(residuals, 1)
    stage_indices = axes(b, 1)
    internal_indices = axes(Phi, 2)
    @assert axes(residuals) == (output_indices,)
    @assert axes(b) == (stage_indices,)
    @assert axes(Phi) == (stage_indices, internal_indices)
    @assert axes(inv_gamma) == (output_indices,)
    @assert axes(selected_indices) == (output_indices,)

    # Construct numeric constants.
    _zero = zero(T)

    # Compute residuals.
    @inbounds for (i, k) in pairs(selected_indices)
        # Compute dot product without SIMD for determinism.
        residual = _zero
        for j in stage_indices
            residual += b[j] * Phi[j, k]
        end
        # Subtract inv_gamma[i] at the end for improved numerical stability.
        residuals[i] = residual - inv_gamma[i]
    end

    return residuals
end


function compute_residual_sum_of_squares(
    b::AbstractVector{T},
    Phi::AbstractMatrix{T},
    inv_gamma::AbstractVector{T},
    selected_indices::AbstractVector{Int},
) where {T}

    # Validate array dimensions.
    stage_indices = axes(b, 1)
    internal_indices = axes(Phi, 2)
    output_indices = axes(inv_gamma, 1)
    @assert axes(b) == (stage_indices,)
    @assert axes(Phi) == (stage_indices, internal_indices)
    @assert axes(inv_gamma) == (output_indices,)
    @assert axes(selected_indices) == (output_indices,)

    # Construct numeric constants.
    _zero = zero(T)

    result = _zero
    @inbounds for (i, k) in pairs(selected_indices)
        # Compute dot product without SIMD for determinism.
        residual = _zero
        for j in stage_indices
            residual += b[j] * Phi[j, k]
        end
        residual -= inv_gamma[i]
        result += abs2(residual)
    end

    return result
end


function compute_residuals!(
    residuals::AbstractVector{T},
    Q::AbstractMatrix{T},
    inv_gamma::AbstractVector{T},
) where {T}

    # Validate array dimensions.
    output_indices = axes(residuals, 1)
    stage_indices = axes(Q, 2)
    @assert axes(residuals) == (output_indices,)
    @assert axes(Q) == (output_indices, stage_indices)
    @assert axes(inv_gamma) == (output_indices,)

    # Construct numeric constants.
    _zero = zero(T)

    # Initialize residuals.
    @simd ivdep for i in output_indices
        @inbounds residuals[i] = -inv_gamma[i]
    end

    # Orthogonalize residuals against columns of Q.
    @inbounds for j in stage_indices
        # Compute dot product without SIMD for determinism.
        overlap = _zero
        for i in output_indices
            overlap += Q[i, j] * residuals[i]
        end
        @simd ivdep for i in output_indices
            residuals[i] -= overlap * Q[i, j]
        end
    end

    return residuals
end


function compute_residuals_and_b!(
    residuals::AbstractVector{T},
    b::AbstractVector{T},
    Q::AbstractMatrix{T},
    inv_gamma::AbstractVector{T},
) where {T}

    # Validate array dimensions.
    output_indices = axes(residuals, 1)
    stage_indices = axes(b, 1)
    @assert axes(residuals) == (output_indices,)
    @assert axes(b) == (stage_indices,)
    @assert axes(Q) == (output_indices, stage_indices)
    @assert axes(inv_gamma) == (output_indices,)

    # Construct numeric constants.
    _zero = zero(T)

    # Initialize residuals.
    @simd ivdep for i in output_indices
        @inbounds residuals[i] = -inv_gamma[i]
    end

    # Orthogonalize residuals against columns of Q.
    @inbounds for j in stage_indices
        # Compute dot product without SIMD for determinism.
        overlap = _zero
        for i in output_indices
            overlap += Q[i, j] * residuals[i]
        end
        @simd ivdep for i in output_indices
            residuals[i] -= overlap * Q[i, j]
        end
        # Store dot product in b.
        b[j] = -overlap
    end

    return (residuals, b)
end


############################################ GRADIENT COMPUTATION (FORWARD-MODE)


"""
    RungeKuttaToolKit.pushforward_dPhi!(
        dPhi::AbstractMatrix{T},
        Phi::AbstractMatrix{T},
        A::AbstractMatrix{T},
        dA::AbstractMatrix{T},
        instructions::AbstractVector{ButcherInstruction},
    ) where {T}

Compute directional derivatives of Butcher weight vectors
``\\{ \\nabla_{\\mathrm{d}A} \\Phi_t(A) : t \\in T \\}``
at a given matrix ``A`` in direction ``\\mathrm{d}A``
over a set of rooted trees ``T`` represented by `instructions`.

# Arguments
- `dPhi`: ``s \\times N`` output matrix. Each directional derivative
    ``\\nabla_{\\mathrm{d}A} \\Phi_t(A)`` is written to `dPhi[:, i]`,
    where ``t`` is the rooted tree represented by `instructions[i]`.
- `Phi`: ``s \\times N`` input matrix containing precomputed
    Butcher weight vectors ``\\Phi_t(A)``. See [`compute_Phi!`](@ref).
- `A`: ``s \\times s`` input matrix containing Runge--Kutta coefficients.
- `dA`: ``s \\times s`` input matrix containing derivative direction.
- `instructions`: length ``N`` input vector of `ButcherInstruction` objects
    encoding a set of rooted trees. See
    [`ButcherInstructions.build_instructions`](@ref).
"""
function pushforward_dPhi!(
    dPhi::AbstractMatrix{T},
    Phi::AbstractMatrix{T},
    A::AbstractMatrix{T},
    dA::AbstractMatrix{T},
    instructions::AbstractVector{ButcherInstruction},
) where {T}

    # Validate array dimensions.
    stage_indices = axes(dPhi, 1)
    internal_indices = axes(dPhi, 2)
    @assert axes(dPhi) == (stage_indices, internal_indices)
    @assert axes(Phi) == (stage_indices, internal_indices)
    @assert axes(A) == (stage_indices, stage_indices)
    @assert axes(dA) == (stage_indices, stage_indices)
    @assert axes(instructions) == (internal_indices,)

    # Construct numeric constants.
    _zero = zero(T)

    # Iterate over Butcher instructions.
    @inbounds for (k, instruction) in pairs(instructions)
        p, q = instruction.left, instruction.right
        if p == NULL_INDEX
            # Trivial tree; derivative is zero.
            @assert q == NULL_INDEX
            @simd ivdep for j in stage_indices
                dPhi[j, k] = _zero
            end
        elseif q == NULL_INDEX
            # Extension; apply product rule to matrix-vector product.
            @simd ivdep for j in stage_indices
                dPhi[j, k] = _zero
            end
            for i in stage_indices
                temp = Phi[i, p]
                dtemp = dPhi[i, p]
                @simd ivdep for j in stage_indices
                    dPhi[j, k] += dA[j, i] * temp + A[j, i] * dtemp
                end
            end
        else
            # Rooted sum; apply product rule to elementwise product.
            @simd ivdep for j in stage_indices
                dPhi[j, k] = dPhi[j, p] * Phi[j, q] + Phi[j, p] * dPhi[j, q]
            end
        end
    end

    return dPhi
end


"""
    RungeKuttaToolKit.pushforward_dPhi!(
        dPhi::AbstractMatrix{T},
        Phi::AbstractMatrix{T},
        A::AbstractMatrix{T},
        u::Int,
        v::Int,
        instructions::AbstractVector{ButcherInstruction},
    ) where {T}

Compute partial derivatives of Butcher weight vectors
``\\{ \\partial_{A_{u,v}} \\Phi_t(A) : t \\in T \\}``
at a given matrix ``A`` with respect to entry ``A_{u,v}``
over a set of rooted trees ``T`` represented by `instructions`.

# Arguments
- `dPhi`: ``s \\times N`` output matrix. Each partial derivative
    ``\\partial_{A_{u,v}} \\Phi_t(A)`` is written to `dPhi[:, i]`,
    where ``t`` is the rooted tree represented by `instructions[i]`.
- `Phi`: ``s \\times N`` input matrix containing precomputed
    Butcher weight vectors ``\\Phi_t(A)``. See [`compute_Phi!`](@ref).
- `A`: ``s \\times s`` input matrix containing Runge--Kutta coefficients.
- `u`: row index of entry with respect to which to differentiate.
- `v`: column index of entry with respect to which to differentiate.
- `instructions`: length ``N`` input vector of `ButcherInstruction` objects
    encoding a set of rooted trees. See
    [`ButcherInstructions.build_instructions`](@ref).
"""
function pushforward_dPhi!(
    dPhi::AbstractMatrix{T},
    Phi::AbstractMatrix{T},
    A::AbstractMatrix{T},
    u::Int,
    v::Int,
    instructions::AbstractVector{ButcherInstruction},
) where {T}

    # Validate array dimensions.
    stage_indices = axes(dPhi, 1)
    internal_indices = axes(dPhi, 2)
    @assert axes(dPhi) == (stage_indices, internal_indices)
    @assert axes(Phi) == (stage_indices, internal_indices)
    @assert axes(A) == (stage_indices, stage_indices)
    @assert u in stage_indices
    @assert v in stage_indices
    @assert axes(instructions) == (internal_indices,)

    # Construct numeric constants.
    _zero = zero(T)

    # Iterate over Butcher instructions.
    @inbounds for (k, instruction) in pairs(instructions)
        p, q = instruction.left, instruction.right
        if p == NULL_INDEX
            # Trivial tree; derivative is zero.
            @assert q == NULL_INDEX
            @simd ivdep for j in stage_indices
                dPhi[j, k] = _zero
            end
        elseif q == NULL_INDEX
            # Extension; apply product rule to matrix-vector product.
            @simd ivdep for j in stage_indices
                dPhi[j, k] = _zero
            end
            dPhi[u, k] = Phi[v, p] # Additional term from product rule.
            for i in stage_indices
                dtemp = dPhi[i, p]
                @simd ivdep for j in stage_indices
                    dPhi[j, k] += dtemp * A[j, i]
                end
            end
        else
            # Rooted sum; apply product rule to elementwise product.
            @simd ivdep for j in stage_indices
                dPhi[j, k] = dPhi[j, p] * Phi[j, q] + Phi[j, p] * dPhi[j, q]
            end
        end
    end

    return dPhi
end


function pushforward_db!(
    db::AbstractVector{T},
    temp::AbstractVector{T},
    dPhi::AbstractMatrix{T},
    b::AbstractVector{T},
    Q::AbstractMatrix{T},
    R::AbstractMatrix{T},
    selected_indices::AbstractVector{Int},
) where {T}

    # Validate array dimensions.
    stage_indices = axes(db, 1)
    output_indices = axes(temp, 1)
    @assert axes(db) == (stage_indices,)
    @assert axes(temp) == (output_indices,)
    @assert axes(dPhi) == (stage_indices, output_indices)
    @assert axes(b) == (stage_indices,)
    @assert axes(Q) == (output_indices, stage_indices)
    @assert axes(R) == (stage_indices, stage_indices)
    @assert axes(selected_indices) == (output_indices,)

    # Construct numeric constants.
    _zero = zero(T)

    @inbounds for (i, k) in pairs(selected_indices)
        # Compute dot product without SIMD for determinism.
        overlap = _zero
        for j in stage_indices
            overlap += b[j] * dPhi[j, k]
        end
        temp[i] = overlap
    end

    @inbounds for i in stage_indices
        # Compute dot product without SIMD for determinism.
        overlap = _zero
        for j in output_indices
            overlap += Q[j, i] * temp[j]
        end
        db[i] = -overlap
    end

    solve_upper_triangular!(db, R)

    return db
end


function pushforward_db!(
    db::AbstractVector{T},
    temp::AbstractVector{T},
    residuals::AbstractVector{T},
    dPhi::AbstractMatrix{T},
    b::AbstractVector{T},
    Q::AbstractMatrix{T},
    R::AbstractMatrix{T},
    selected_indices::AbstractVector{Int},
) where {T}

    # Validate array dimensions.
    stage_indices = axes(db, 1)
    output_indices = axes(temp, 1)
    internal_indices = axes(dPhi, 2)
    @assert axes(db) == (stage_indices,)
    @assert axes(temp) == (output_indices,)
    @assert axes(residuals) == (output_indices,)
    @assert axes(dPhi) == (stage_indices, internal_indices)
    @assert axes(b) == (stage_indices,)
    @assert axes(Q) == (output_indices, stage_indices)
    @assert axes(R) == (stage_indices, stage_indices)
    @assert axes(selected_indices) == (output_indices,)

    # Construct numeric constants.
    _zero = zero(T)

    @inbounds begin

        @simd ivdep for i in stage_indices
            db[i] = _zero
        end

        for (i, k) in pairs(selected_indices)
            r = residuals[i]
            @simd ivdep for j in stage_indices
                db[j] += r * dPhi[j, k]
            end
        end

    end

    solve_lower_triangular!(db, R)

    @inbounds begin

        for (i, k) in pairs(selected_indices)
            # Compute dot product without SIMD for determinism.
            overlap = _zero
            for j in stage_indices
                overlap += b[j] * dPhi[j, k]
            end
            temp[i] = overlap
        end

        for i in stage_indices
            # Compute dot product without SIMD for determinism.
            overlap = _zero
            for j in output_indices
                overlap += Q[j, i] * temp[j]
            end
            db[i] = -(db[i] + overlap)
        end

    end

    solve_upper_triangular!(db, R)

    return db
end


function pushforward_dresiduals!(
    dresiduals::AbstractVector{T},
    b::AbstractVector{T},
    dPhi::AbstractMatrix{T},
    selected_indices::AbstractVector{Int},
) where {T}

    # Validate array dimensions.
    output_indices = axes(dresiduals, 1)
    stage_indices = axes(b, 1)
    internal_indices = axes(dPhi, 2)
    @assert axes(dresiduals) == (output_indices,)
    @assert axes(b) == (stage_indices,)
    @assert axes(dPhi) == (stage_indices, internal_indices)
    @assert axes(selected_indices) == (output_indices,)

    # Construct numeric constants.
    _zero = zero(T)

    @inbounds for (i, k) in pairs(selected_indices)
        # Compute dot product without SIMD for determinism.
        doverlap = _zero
        for j in stage_indices
            doverlap += b[j] * dPhi[j, k]
        end
        dresiduals[i] = doverlap
    end

    return dresiduals
end


function pushforward_dresiduals!(
    dresiduals::AbstractVector{T},
    db::AbstractVector{T},
    b::AbstractVector{T},
    dPhi::AbstractMatrix{T},
    Phi::AbstractMatrix{T},
    selected_indices::AbstractVector{Int},
) where {T}

    # Validate array dimensions.
    output_indices = axes(dresiduals, 1)
    stage_indices = axes(db, 1)
    internal_indices = axes(dPhi, 2)
    @assert axes(dresiduals) == (output_indices,)
    @assert axes(db) == (stage_indices,)
    @assert axes(b) == (stage_indices,)
    @assert axes(dPhi) == (stage_indices, internal_indices)
    @assert axes(Phi) == (stage_indices, internal_indices)
    @assert axes(selected_indices) == (output_indices,)

    # Construct numeric constants.
    _zero = zero(T)

    @inbounds for (i, k) in pairs(selected_indices)
        # Compute dot product without SIMD for determinism.
        doverlap = _zero
        for j in stage_indices
            doverlap += db[j] * Phi[j, k] + b[j] * dPhi[j, k]
        end
        dresiduals[i] = doverlap
    end

    return dresiduals
end


############################################ GRADIENT COMPUTATION (REVERSE-MODE)


function pullback_dPhi_from_residual!(
    dPhi::AbstractMatrix{T},
    b::AbstractVector{T},
    residuals::AbstractVector{T},
    source_indices::AbstractVector{Int},
) where {T}

    # Validate array dimensions.
    stage_indices = axes(dPhi, 1)
    internal_indices = axes(dPhi, 2)
    output_indices = axes(residuals, 1)
    @assert axes(dPhi) == (stage_indices, internal_indices)
    @assert axes(b) == (stage_indices,)
    @assert axes(residuals) == (output_indices,)
    @assert axes(source_indices) == (internal_indices,)

    # Construct numeric constants.
    _zero = zero(T)

    @inbounds for (k, s) in Iterators.reverse(pairs(source_indices))
        if s == NULL_INDEX
            @simd ivdep for j in stage_indices
                dPhi[j, k] = _zero
            end
        else
            residual = residuals[s]
            twice_residual = residual + residual
            @simd ivdep for j in stage_indices
                dPhi[j, k] = twice_residual * b[j]
            end
        end
    end

    return dPhi
end


function pullback_dPhi_initial!(
    dPhi::AbstractMatrix{T},
    b::AbstractVector{T},
    Phi::AbstractMatrix{T},
    inv_gamma::AbstractVector{T},
    source_indices::AbstractVector{Int},
) where {T}

    # Validate array dimensions.
    stage_indices = axes(dPhi, 1)
    internal_indices = axes(dPhi, 2)
    output_indices = axes(inv_gamma, 1)
    @assert axes(dPhi) == (stage_indices, internal_indices)
    @assert axes(b) == (stage_indices,)
    @assert axes(Phi) == (stage_indices, internal_indices)
    @assert axes(inv_gamma) == (output_indices,)
    @assert axes(source_indices) == (internal_indices,)

    # Construct numeric constants.
    _zero = zero(T)

    @inbounds for (k, s) in Iterators.reverse(pairs(source_indices))
        if s == NULL_INDEX
            @simd ivdep for j in stage_indices
                dPhi[j, k] = _zero
            end
        else
            # Compute dot product without SIMD for determinism.
            residual = _zero
            for j in stage_indices
                residual += b[j] * Phi[j, k]
            end
            # Subtract inv_gamma[s] at the end for improved numerical stability.
            residual -= inv_gamma[s]
            # Double residual. (Addition is faster than multiplication by two.)
            twice_residual = residual + residual
            @simd ivdep for j in stage_indices
                dPhi[j, k] = twice_residual * b[j]
            end
        end
    end

    return dPhi
end


function pullback_dPhi!(
    dPhi::AbstractMatrix{T},
    A::AbstractMatrix{T},
    Phi::AbstractMatrix{T},
    extension_indices::AbstractVector{Int},
    rooted_sum_ranges::AbstractVector{UnitRange{Int}},
    rooted_sum_indices::AbstractVector{Pair{Int,Int}},
) where {T}

    # Validate array dimensions.
    stage_indices = axes(dPhi, 1)
    internal_indices = axes(dPhi, 2)
    @assert axes(dPhi) == (stage_indices, internal_indices)
    @assert axes(A) == (stage_indices, stage_indices)
    @assert axes(Phi) == (stage_indices, internal_indices)
    @assert axes(extension_indices) == (internal_indices,)
    @assert axes(rooted_sum_ranges) == (internal_indices,)

    @inbounds for k in Iterators.reverse(internal_indices)
        c = extension_indices[k]
        if c != NULL_INDEX
            for j in stage_indices
                temp = dPhi[j, k]
                for i in stage_indices
                    temp += A[i, j] * dPhi[i, c]
                end
                dPhi[j, k] = temp
            end
        end
        for i in rooted_sum_ranges[k]
            (p, q) = rooted_sum_indices[i]
            @simd ivdep for j in stage_indices
                dPhi[j, k] += Phi[j, p] * dPhi[j, q]
            end
        end
    end

    return dPhi
end


function pullback_dA!(
    dA::AbstractMatrix{T},
    Phi::AbstractMatrix{T},
    dPhi::AbstractMatrix{T},
    extension_indices::AbstractVector{Int},
) where {T}

    # Validate array dimensions.
    stage_indices = axes(dA, 1)
    internal_indices = axes(dPhi, 2)
    @assert axes(dA) == (stage_indices, stage_indices)
    @assert axes(Phi) == (stage_indices, internal_indices)
    @assert axes(dPhi) == (stage_indices, internal_indices)
    @assert axes(extension_indices) == (internal_indices,)

    # Construct numeric constants.
    _zero = zero(T)

    # Initialize dA to zero.
    @inbounds for j in stage_indices
        @simd ivdep for i in stage_indices
            dA[i, j] = _zero
        end
    end

    # Iterate over intermediate trees obtained by extension.
    @inbounds for (k, c) in pairs(extension_indices)
        if c != NULL_INDEX
            for t in stage_indices
                f = Phi[t, k]
                @simd ivdep for s in stage_indices
                    dA[s, t] += f * dPhi[s, c]
                end
            end
        end
    end

    return dA
end


function pullback_db!(
    db::AbstractVector{T},
    Phi::AbstractMatrix{T},
    residuals::AbstractVector{T},
    selected_indices::AbstractVector{Int},
) where {T}

    # Validate array dimensions.
    stage_indices = axes(db, 1)
    internal_indices = axes(Phi, 2)
    output_indices = axes(residuals, 1)
    @assert axes(db) == (stage_indices,)
    @assert axes(Phi) == (stage_indices, internal_indices)
    @assert axes(residuals) == (output_indices,)
    @assert axes(selected_indices) == (output_indices,)

    # Construct numeric constants.
    _zero = zero(T)

    @inbounds begin

        # Initialize db to zero.
        @simd ivdep for i in stage_indices
            db[i] = _zero
        end

        for (i, k) in pairs(selected_indices)
            r = residuals[i]
            @simd ivdep for j in stage_indices
                db[j] += r * Phi[j, k]
            end
        end

        # Double db. (Addition is faster than multiplication by two.)
        @simd ivdep for i in stage_indices
            db[i] += db[i]
        end

    end

    return db
end


function pullback_db!(
    db::AbstractVector{T},
    b::AbstractVector{T},
    Phi::AbstractMatrix{T},
    inv_gamma::AbstractVector{T},
    selected_indices::AbstractVector{Int},
) where {T}

    # Validate array dimensions.
    stage_indices = axes(db, 1)
    internal_indices = axes(Phi, 2)
    output_indices = axes(inv_gamma, 1)
    @assert axes(db) == (stage_indices,)
    @assert axes(b) == (stage_indices,)
    @assert axes(Phi) == (stage_indices, internal_indices)
    @assert axes(inv_gamma) == (output_indices,)
    @assert axes(selected_indices) == (output_indices,)

    # Construct numeric constants.
    _zero = zero(T)

    @inbounds begin

        # Initialize db to zero.
        @simd ivdep for i in stage_indices
            db[i] = _zero
        end

        for (i, k) in pairs(selected_indices)
            # Compute dot product without SIMD for determinism.
            residual = _zero
            for j in stage_indices
                residual += b[j] * Phi[j, k]
            end
            # Subtract inv_gamma[i] at the end for improved numerical stability.
            residual -= inv_gamma[i]
            @simd ivdep for j in stage_indices
                db[j] += residual * Phi[j, k]
            end
        end

        # Double db. (Addition is faster than multiplication by two.)
        @simd ivdep for i in stage_indices
            db[i] += db[i]
        end

    end

    return db
end


########################################################### LEAST-SQUARES SOLVER


@inline inv_sqrt(x::Float16) = rsqrt(x)
@inline inv_sqrt(x::Float32) = rsqrt(x)
@inline inv_sqrt(x::Float64) = rsqrt(x)
@inline inv_sqrt(x::MultiFloat{T,N}) where {T,N} = rsqrt(x)
@inline inv_sqrt(x::T) where {T} = inv(sqrt(x))


# function gram_schmidt_qr!(Q::AbstractMatrix{T}) where {T}
#     t, s = size(Q)
#     _zero = zero(T)
#     @inbounds for i = 1:s
#         squared_norm = _zero
#         for k = 1:t
#             squared_norm += abs2(Q[k, i])
#         end
#         if !iszero(squared_norm)
#             inv_norm = inv_sqrt(squared_norm)
#             @simd ivdep for k = 1:t
#                 Q[k, i] *= inv_norm
#             end
#             for j = i+1:s
#                 overlap = _zero
#                 for k = 1:t
#                     overlap += Q[k, i] * Q[k, j]
#                 end
#                 @simd ivdep for k = 1:t
#                     Q[k, j] -= overlap * Q[k, i]
#                 end
#             end
#         end
#     end
#     return Q
# end


# function gram_schmidt_qr!(Q::AbstractMatrix{T}, R::AbstractMatrix{T}) where {T}
#     t, s = size(Q)
#     @assert (s, s) == size(R)
#     # NOTE: R is stored transposed, and its diagonal is stored inverted.
#     _zero = zero(T)
#     @inbounds for i = 1:s
#         squared_norm = _zero
#         for k = 1:t
#             squared_norm += abs2(Q[k, i])
#         end
#         if !iszero(squared_norm)
#             inv_norm = inv_sqrt(squared_norm)
#             R[i, i] = inv_norm
#             @simd ivdep for k = 1:t
#                 Q[k, i] *= inv_norm
#             end
#             for j = i+1:s
#                 overlap = _zero
#                 for k = 1:t
#                     overlap += Q[k, i] * Q[k, j]
#                 end
#                 R[j, i] = overlap
#                 @simd ivdep for k = 1:t
#                     Q[k, j] -= overlap * Q[k, i]
#                 end
#             end
#         else
#             R[i, i] = _zero
#         end
#     end
#     return (Q, R)
# end


# function solve_upper_triangular!(
#     b::AbstractVector{T}, R::AbstractMatrix{T}
# ) where {T}
#     s = length(b)
#     @assert (s, s) == size(R)
#     # NOTE: R is stored transposed, and its diagonal is stored inverted.
#     _zero = zero(T)
#     @inbounds for i = s:-1:1
#         if iszero(R[i, i])
#             b[i] = _zero
#         else
#             overlap = _zero
#             for j = i+1:s
#                 overlap += R[j, i] * b[j]
#             end
#             b[i] = R[i, i] * (b[i] - overlap)
#         end
#     end
#     return b
# end


# function solve_lower_triangular!(
#     b::AbstractVector{T}, L::AbstractMatrix{T}
# ) where {T}
#     s = length(b)
#     @assert (s, s) == size(L)
#     # NOTE: The diagonal of L is stored inverted.
#     _zero = zero(T)
#     @inbounds for i = 1:s
#         if iszero(L[i, i])
#             b[i] = _zero
#         else
#             b[i] *= L[i, i]
#             for j = i+1:s
#                 b[j] -= L[j, i] * b[i]
#             end
#         end
#     end
#     return b
# end


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


# function residual_norm_squared(residuals::AbstractVector{T}) where {T}
#     result = zero(T)
#     for r in residuals
#         result += abs2(r)
#     end
#     return result
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


# function (ev::RKOCEvaluatorBE{T})(x::Vector{T}) where {T}
#     reshape_explicit!(ev.A, ev.b, x)
#     compute_Phi!(ev.Phi, ev.A, ev.table.instructions)
#     compute_residuals!(ev.residuals,
#         ev.b, ev.Phi, ev.inv_gamma, ev.table.selected_indices)
#     return residual_norm_squared(ev.residuals)
# end


# function (ev::RKOCEvaluatorBD{T})(x::Vector{T}) where {T}
#     reshape_diagonally_implicit!(ev.A, ev.b, x)
#     compute_Phi!(ev.Phi, ev.A, ev.table.instructions)
#     compute_residuals!(ev.residuals,
#         ev.b, ev.Phi, ev.inv_gamma, ev.table.selected_indices)
#     return residual_norm_squared(ev.residuals)
# end


# function (ev::RKOCEvaluatorBI{T})(x::Vector{T}) where {T}
#     reshape_implicit!(ev.A, ev.b, x)
#     compute_Phi!(ev.Phi, ev.A, ev.table.instructions)
#     compute_residuals!(ev.residuals,
#         ev.b, ev.Phi, ev.inv_gamma, ev.table.selected_indices)
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


# function (ev::RKOCResidualEvaluatorBE{T})(
#     residuals::Vector{T}, x::Vector{T}
# ) where {T}
#     reshape_explicit!(ev.A, ev.b, x)
#     compute_Phi!(ev.Phi, ev.A, ev.table.instructions)
#     compute_residuals!(residuals,
#         ev.b, ev.Phi, ev.inv_gamma, ev.table.selected_indices)
#     return residuals
# end


# function (ev::RKOCResidualEvaluatorBD{T})(
#     residuals::Vector{T}, x::Vector{T}
# ) where {T}
#     reshape_diagonally_implicit!(ev.A, ev.b, x)
#     compute_Phi!(ev.Phi, ev.A, ev.table.instructions)
#     compute_residuals!(residuals,
#         ev.b, ev.Phi, ev.inv_gamma, ev.table.selected_indices)
#     return residuals
# end


# function (ev::RKOCResidualEvaluatorBI{T})(
#     residuals::Vector{T}, x::Vector{T}
# ) where {T}
#     reshape_implicit!(ev.A, ev.b, x)
#     compute_Phi!(ev.Phi, ev.A, ev.table.instructions)
#     compute_residuals!(residuals,
#         ev.b, ev.Phi, ev.inv_gamma, ev.table.selected_indices)
#     return residuals
# end


# @inline (ev::RKOCResidualEvaluatorAE{T})(x::Vector{T}) where {T} =
#     ev(Vector{T}(undef, length(ev.table.selected_indices)), x)
# @inline (ev::RKOCResidualEvaluatorAD{T})(x::Vector{T}) where {T} =
#     ev(Vector{T}(undef, length(ev.table.selected_indices)), x)
# @inline (ev::RKOCResidualEvaluatorAI{T})(x::Vector{T}) where {T} =
#     ev(Vector{T}(undef, length(ev.table.selected_indices)), x)
# @inline (ev::RKOCResidualEvaluatorBE{T})(x::Vector{T}) where {T} =
#     ev(Vector{T}(undef, length(ev.table.selected_indices)), x)
# @inline (ev::RKOCResidualEvaluatorBD{T})(x::Vector{T}) where {T} =
#     ev(Vector{T}(undef, length(ev.table.selected_indices)), x)
# @inline (ev::RKOCResidualEvaluatorBI{T})(x::Vector{T}) where {T} =
#     ev(Vector{T}(undef, length(ev.table.selected_indices)), x)


# struct RKOCEvaluatorAEAdjoint{T}
#     ev::RKOCEvaluatorAE{T}
# end
# struct RKOCEvaluatorADAdjoint{T}
#     ev::RKOCEvaluatorAD{T}
# end
# struct RKOCEvaluatorAIAdjoint{T}
#     ev::RKOCEvaluatorAI{T}
# end
# struct RKOCEvaluatorBEAdjoint{T}
#     ev::RKOCEvaluatorBE{T}
# end
# struct RKOCEvaluatorBDAdjoint{T}
#     ev::RKOCEvaluatorBD{T}
# end
# struct RKOCEvaluatorBIAdjoint{T}
#     ev::RKOCEvaluatorBI{T}
# end
# struct RKOCResidualEvaluatorAEAdjoint{T}
#     ev::RKOCResidualEvaluatorAE{T}
# end
# struct RKOCResidualEvaluatorADAdjoint{T}
#     ev::RKOCResidualEvaluatorAD{T}
# end
# struct RKOCResidualEvaluatorAIAdjoint{T}
#     ev::RKOCResidualEvaluatorAI{T}
# end
# struct RKOCResidualEvaluatorBEAdjoint{T}
#     ev::RKOCResidualEvaluatorBE{T}
# end
# struct RKOCResidualEvaluatorBDAdjoint{T}
#     ev::RKOCResidualEvaluatorBD{T}
# end
# struct RKOCResidualEvaluatorBIAdjoint{T}
#     ev::RKOCResidualEvaluatorBI{T}
# end


# @inline Base.adjoint(ev::RKOCEvaluatorAE{T}) where {T} =
#     RKOCEvaluatorAEAdjoint{T}(ev)
# @inline Base.adjoint(ev::RKOCEvaluatorAD{T}) where {T} =
#     RKOCEvaluatorADAdjoint{T}(ev)
# @inline Base.adjoint(ev::RKOCEvaluatorAI{T}) where {T} =
#     RKOCEvaluatorAIAdjoint{T}(ev)
# @inline Base.adjoint(ev::RKOCEvaluatorBE{T}) where {T} =
#     RKOCEvaluatorBEAdjoint{T}(ev)
# @inline Base.adjoint(ev::RKOCEvaluatorBD{T}) where {T} =
#     RKOCEvaluatorBDAdjoint{T}(ev)
# @inline Base.adjoint(ev::RKOCEvaluatorBI{T}) where {T} =
#     RKOCEvaluatorBIAdjoint{T}(ev)
# @inline Base.adjoint(ev::RKOCResidualEvaluatorAE{T}) where {T} =
#     RKOCResidualEvaluatorAEAdjoint{T}(ev)
# @inline Base.adjoint(ev::RKOCResidualEvaluatorAD{T}) where {T} =
#     RKOCResidualEvaluatorADAdjoint{T}(ev)
# @inline Base.adjoint(ev::RKOCResidualEvaluatorAI{T}) where {T} =
#     RKOCResidualEvaluatorAIAdjoint{T}(ev)
# @inline Base.adjoint(ev::RKOCResidualEvaluatorBE{T}) where {T} =
#     RKOCResidualEvaluatorBEAdjoint{T}(ev)
# @inline Base.adjoint(ev::RKOCResidualEvaluatorBD{T}) where {T} =
#     RKOCResidualEvaluatorBDAdjoint{T}(ev)
# @inline Base.adjoint(ev::RKOCResidualEvaluatorBI{T}) where {T} =
#     RKOCResidualEvaluatorBIAdjoint{T}(ev)


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


# function (adj::RKOCEvaluatorBEAdjoint{T})(g::Vector{T}, x::Vector{T}) where {T}
#     reshape_explicit!(adj.ev.A, adj.ev.b, x)
#     compute_Phi!(adj.ev.Phi, adj.ev.A, adj.ev.table.instructions)
#     compute_residuals!(adj.ev.residuals,
#         adj.ev.b, adj.ev.Phi, adj.ev.inv_gamma, adj.ev.table.selected_indices)
#     pullback_dPhi_from_residual!(adj.ev.dPhi,
#         adj.ev.b, adj.ev.residuals, adj.ev.table.source_indices)
#     pullback_dPhi!(adj.ev.dPhi,
#         adj.ev.A, adj.ev.Phi, adj.ev.table.extension_indices,
#         adj.ev.table.rooted_sum_ranges, adj.ev.table.rooted_sum_indices)
#     pullback_dA!(adj.ev.dA,
#         adj.ev.Phi, adj.ev.dPhi, adj.ev.table.extension_indices)
#     pullback_db!(adj.ev.db,
#         adj.ev.Phi, adj.ev.residuals, adj.ev.table.selected_indices)
#     reshape_explicit!(g, adj.ev.dA, adj.ev.db)
#     return g
# end


# function (adj::RKOCEvaluatorBDAdjoint{T})(g::Vector{T}, x::Vector{T}) where {T}
#     reshape_diagonally_implicit!(adj.ev.A, adj.ev.b, x)
#     compute_Phi!(adj.ev.Phi, adj.ev.A, adj.ev.table.instructions)
#     compute_residuals!(adj.ev.residuals,
#         adj.ev.b, adj.ev.Phi, adj.ev.inv_gamma, adj.ev.table.selected_indices)
#     pullback_dPhi_from_residual!(adj.ev.dPhi,
#         adj.ev.b, adj.ev.residuals, adj.ev.table.source_indices)
#     pullback_dPhi!(adj.ev.dPhi,
#         adj.ev.A, adj.ev.Phi, adj.ev.table.extension_indices,
#         adj.ev.table.rooted_sum_ranges, adj.ev.table.rooted_sum_indices)
#     pullback_dA!(adj.ev.dA,
#         adj.ev.Phi, adj.ev.dPhi, adj.ev.table.extension_indices)
#     pullback_db!(adj.ev.db,
#         adj.ev.Phi, adj.ev.residuals, adj.ev.table.selected_indices)
#     reshape_diagonally_implicit!(g, adj.ev.dA, adj.ev.db)
#     return g
# end


# function (adj::RKOCEvaluatorBIAdjoint{T})(g::Vector{T}, x::Vector{T}) where {T}
#     reshape_implicit!(adj.ev.A, adj.ev.b, x)
#     compute_Phi!(adj.ev.Phi, adj.ev.A, adj.ev.table.instructions)
#     compute_residuals!(adj.ev.residuals,
#         adj.ev.b, adj.ev.Phi, adj.ev.inv_gamma, adj.ev.table.selected_indices)
#     pullback_dPhi_from_residual!(adj.ev.dPhi,
#         adj.ev.b, adj.ev.residuals, adj.ev.table.source_indices)
#     pullback_dPhi!(adj.ev.dPhi,
#         adj.ev.A, adj.ev.Phi, adj.ev.table.extension_indices,
#         adj.ev.table.rooted_sum_ranges, adj.ev.table.rooted_sum_indices)
#     pullback_dA!(adj.ev.dA,
#         adj.ev.Phi, adj.ev.dPhi, adj.ev.table.extension_indices)
#     pullback_db!(adj.ev.db,
#         adj.ev.Phi, adj.ev.residuals, adj.ev.table.selected_indices)
#     reshape_implicit!(g, adj.ev.dA, adj.ev.db)
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
#     for j = 1:s
#         for (i, m) in pairs(adj.ev.table.selected_indices)
#             @inbounds jacobian[i, k] = adj.ev.Phi[j, m]
#         end
#         k += 1
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


# @inline (adj::RKOCEvaluatorAEAdjoint{T})(x::Vector{T}) where {T} =
#     adj(similar(x), x)
# @inline (adj::RKOCEvaluatorADAdjoint{T})(x::Vector{T}) where {T} =
#     adj(similar(x), x)
# @inline (adj::RKOCEvaluatorAIAdjoint{T})(x::Vector{T}) where {T} =
#     adj(similar(x), x)
# @inline (adj::RKOCEvaluatorBEAdjoint{T})(x::Vector{T}) where {T} =
#     adj(similar(x), x)
# @inline (adj::RKOCEvaluatorBDAdjoint{T})(x::Vector{T}) where {T} =
#     adj(similar(x), x)
# @inline (adj::RKOCEvaluatorBIAdjoint{T})(x::Vector{T}) where {T} =
#     adj(similar(x), x)
# @inline (adj::RKOCResidualEvaluatorAEAdjoint{T})(x::Vector{T}) where {T} =
#     adj(Matrix{T}(undef, length(adj.ev.table.selected_indices), length(x)), x)
# @inline (adj::RKOCResidualEvaluatorADAdjoint{T})(x::Vector{T}) where {T} =
#     adj(Matrix{T}(undef, length(adj.ev.table.selected_indices), length(x)), x)
# @inline (adj::RKOCResidualEvaluatorAIAdjoint{T})(x::Vector{T}) where {T} =
#     adj(Matrix{T}(undef, length(adj.ev.table.selected_indices), length(x)), x)
# @inline (adj::RKOCResidualEvaluatorBEAdjoint{T})(x::Vector{T}) where {T} =
#     adj(Matrix{T}(undef, length(adj.ev.table.selected_indices), length(x)), x)
# @inline (adj::RKOCResidualEvaluatorBDAdjoint{T})(x::Vector{T}) where {T} =
#     adj(Matrix{T}(undef, length(adj.ev.table.selected_indices), length(x)), x)
# @inline (adj::RKOCResidualEvaluatorBIAdjoint{T})(x::Vector{T}) where {T} =
#     adj(Matrix{T}(undef, length(adj.ev.table.selected_indices), length(x)), x)


end # module RungeKuttaToolKit
