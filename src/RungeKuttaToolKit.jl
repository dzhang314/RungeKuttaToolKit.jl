module RungeKuttaToolKit


include("ExampleMethods.jl")


const NULL_INDEX = typemin(Int)
include("ButcherInstructions.jl")


const PERFORM_INTERNAL_BOUNDS_CHECKS = true
abstract type AbstractRKOCEvaluator{T} end
abstract type AbstractRKOCEvaluatorAO{T} end
abstract type AbstractRKParameterization{T} end
abstract type AbstractRKParameterizationAO{T} end
function get_axes end
function compute_Phi! end
function pushforward_dPhi! end
function pushforward_dresiduals! end
include("RKParameterization.jl")


abstract type AbstractRKOCAdjoint{T} end
abstract type AbstractRKOCAdjointAO{T} end
abstract type AbstractRKCost{T} end
function compute_residual end
include("RKCost.jl")
using .RKCost


function compute_residuals! end
function pullback_dPhi! end
function pullback_dA! end
include("AbstractRKInterface.jl")


using .ButcherInstructions: LevelSequence,
    NULL_INDEX, ButcherInstruction, ButcherInstructionTable,
    rooted_trees, all_rooted_trees, butcher_density, butcher_symmetry
export LevelSequence, ButcherInstruction, ButcherInstructionTable,
    rooted_trees, all_rooted_trees, butcher_density, butcher_symmetry


using SIMD: Vec, vload, vstore, vgather


####################################################### EVALUATOR DATA STRUCTURE


export RKOCEvaluator, RKOCEvaluatorSIMD, RKOCEvaluatorAO


struct RKOCEvaluator{T} <: AbstractRKOCEvaluator{T}
    table::ButcherInstructionTable
    Phi::Matrix{T}
    dPhi::Matrix{T}
    inv_gamma::Vector{T}
end


struct RKOCEvaluatorAO{T} <: AbstractRKOCEvaluatorAO{T}
    table::ButcherInstructionTable
    Phi::Matrix{T}
    dPhi::Matrix{T}
    Q::Matrix{T}
    R::Matrix{T}
    inv_gamma::Vector{T}
end


"""
    RKOCEvaluator{T}(
        trees::AbstractVector{LevelSequence},
        num_stages::Integer,
    ) -> RKOCEvaluator{T}

Construct an `RKOCEvaluator` that encodes a given list of rooted trees.

# Arguments
- `trees`: input list of rooted trees in `LevelSequence` representation.
- `num_stages`: number of stages (i.e., size of the Butcher tableau). Must be
    specified at construction time to allocate internal workspace arrays.

If the type `T` is not specified, it defaults to `Float64`.
"""
function RKOCEvaluator{T}(
    trees::AbstractVector{LevelSequence},
    num_stages::Integer,
) where {T}
    table = ButcherInstructionTable(trees)
    return RKOCEvaluator{T}(table,
        Matrix{T}(undef, num_stages, length(table.instructions)),
        Matrix{T}(undef, num_stages, length(table.instructions)),
        [inv(T(butcher_density(tree))) for tree in trees])
end


RKOCEvaluator(trees::AbstractVector{LevelSequence}, num_stages::Integer) =
    RKOCEvaluator{Float64}(trees, num_stages)


"""
    RKOCEvaluator{T}(order::Integer, num_stages::Integer) -> RKOCEvaluator{T}

Construct an `RKOCEvaluator` that encodes all rooted trees
having at most `order` vertices. This is equivalent to
`RKOCEvaluator{T}(all_rooted_trees(order), num_stages)`.

If the type `T` is not specified, it defaults to `Float64`.
"""
RKOCEvaluator{T}(order::Integer, num_stages::Integer) where {T} =
    RKOCEvaluator{T}(all_rooted_trees(order), num_stages)


RKOCEvaluator(order::Integer, num_stages::Integer) =
    RKOCEvaluator{Float64}(order, num_stages)


struct RKOCEvaluatorSIMD{S,T} <: AbstractRKOCEvaluator{T}
    table::ButcherInstructionTable
    Phi::Vector{Vec{S,T}}
    dPhi::Vector{Vec{S,T}}
    inv_gamma::Vector{T}
end


"""
    RKOCEvaluatorSIMD{S,T}(
        trees::AbstractVector{LevelSequence},
    ) -> RKOCEvaluatorSIMD{S,T}

Construct an `RKOCEvaluatorSIMD` that encodes a given list of rooted trees.

Unlike a standard `RKOCEvaluator`, the number of stages `S` is provided as a
static type parameter in order to statically determine the SIMD vector width.

# Arguments
- `trees`: input list of rooted trees in `LevelSequence` representation.

If the type `T` is not specified, it defaults to `Float64`.
"""
function RKOCEvaluatorSIMD{S,T}(
    trees::AbstractVector{LevelSequence},
) where {S,T}
    table = ButcherInstructionTable(trees)
    return RKOCEvaluatorSIMD{S,T}(table,
        Vector{Vec{S,T}}(undef, length(table.instructions)),
        Vector{Vec{S,T}}(undef, length(table.instructions)),
        [inv(T(butcher_density(tree))) for tree in trees])
end


RKOCEvaluatorSIMD{S}(trees::AbstractVector{LevelSequence}) where {S} =
    RKOCEvaluatorSIMD{S,Float64}(trees)


"""
    RKOCEvaluatorSIMD{S,T}(order::Integer) -> RKOCEvaluatorSIMD{S,T}

Construct an `RKOCEvaluatorSIMD` that encodes all rooted trees
having at most `order` vertices. This is equivalent to
`RKOCEvaluatorSIMD{S,T}(all_rooted_trees(order))`.

Unlike a standard `RKOCEvaluator`, the number of stages `S` is provided as a
static type parameter in order to statically determine the SIMD vector width.

If the type `T` is not specified, it defaults to `Float64`.
"""
RKOCEvaluatorSIMD{S,T}(order::Integer) where {S,T} =
    RKOCEvaluatorSIMD{S,T}(all_rooted_trees(order))


RKOCEvaluatorSIMD{S}(order::Integer) where {S} =
    RKOCEvaluatorSIMD{S,Float64}(order)


################################################################################


struct RKOCAdjoint{T,E} <: AbstractRKOCAdjoint{T}
    ev::E
end


@inline Base.adjoint(ev::E) where {T,E<:AbstractRKOCEvaluator{T}} =
    RKOCAdjoint{T,E}(ev)


################################################################################


@inline function get_axes(ev::RKOCEvaluator{T}) where {T}
    stage_axis = axes(ev.Phi, 1)
    internal_axis = axes(ev.Phi, 2)
    output_axis = axes(ev.inv_gamma, 1)
    @static if PERFORM_INTERNAL_BOUNDS_CHECKS
        @assert axes(ev.table.instructions) == (internal_axis,)
        @assert axes(ev.table.selected_indices) == (output_axis,)
        @assert axes(ev.table.source_indices) == (internal_axis,)
        @assert axes(ev.table.extension_indices) == (internal_axis,)
        @assert axes(ev.table.rooted_sum_ranges) == (internal_axis,)
        @assert axes(ev.Phi) == (stage_axis, internal_axis)
        @assert axes(ev.dPhi) == (stage_axis, internal_axis)
        @assert axes(ev.inv_gamma) == (output_axis,)
    end
    return (stage_axis, internal_axis, output_axis)
end


@inline function get_axes(ev::RKOCEvaluatorSIMD{S,T}) where {S,T}
    stage_axis = Base.OneTo(S)
    internal_axis = axes(ev.Phi, 1)
    output_axis = axes(ev.inv_gamma, 1)
    @static if PERFORM_INTERNAL_BOUNDS_CHECKS
        @assert axes(ev.table.instructions) == (internal_axis,)
        @assert axes(ev.table.selected_indices) == (output_axis,)
        @assert axes(ev.table.source_indices) == (internal_axis,)
        @assert axes(ev.table.extension_indices) == (internal_axis,)
        @assert axes(ev.table.rooted_sum_ranges) == (internal_axis,)
        @assert axes(ev.Phi) == (internal_axis,)
        @assert axes(ev.dPhi) == (internal_axis,)
        @assert axes(ev.inv_gamma) == (output_axis,)
    end
    return (stage_axis, internal_axis, output_axis)
end


@inline function vload_vec(::Val{N}, v::Vector{T}) where {N,T}
    @static if PERFORM_INTERNAL_BOUNDS_CHECKS
        @assert size(v) == (N,)
    end
    ptr = pointer(v)
    return vload(Vec{N,T}, ptr)
end


@inline function vstore_vec!(v::Vector{T}, data::Vec{N,T}) where {N,T}
    @static if PERFORM_INTERNAL_BOUNDS_CHECKS
        @assert size(v) == (N,)
    end
    ptr = pointer(v)
    vstore(data, ptr, nothing)
    return v
end


@inline function vload_cols(::Val{N}, A::Matrix{T}) where {N,T}
    @static if PERFORM_INTERNAL_BOUNDS_CHECKS
        @assert size(A) == (N, N)
    end
    row_size = N * sizeof(T)
    ptr = pointer(A)
    return ntuple(
        i -> vload(Vec{N,T}, ptr + (i - 1) * row_size),
        Val{N}())
end


@inline function vload_rows(::Val{N}, A::Matrix{T}) where {N,T}
    @static if PERFORM_INTERNAL_BOUNDS_CHECKS
        @assert size(A) == (N, N)
    end
    entry_size = sizeof(T)
    row_size = N * entry_size
    iota = Vec{N,Int}(ntuple(i -> (i - 1) * row_size, Val{N}()))
    base = pointer(A) + iota
    return ntuple(
        i -> vgather(base + (i - 1) * entry_size),
        Val{N}())
end


@inline function vsum(v::Vec{N,T}) where {N,T}
    # Compute dot products sequentially for determinism.
    # SIMD parallel sum instructions can change summation order.
    result = zero(T)
    for i = 1:N
        result += v[i]
    end
    return result
end


################################################### PHI AND RESIDUAL COMPUTATION


function compute_Phi!(
    ev::RKOCEvaluator{T},
    A::AbstractMatrix{T},
) where {T}

    # Validate array dimensions.
    stage_axis, _, _ = get_axes(ev)
    @static if PERFORM_INTERNAL_BOUNDS_CHECKS
        @assert axes(A) == (stage_axis, stage_axis)
    end

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


function compute_Phi!(
    ev::RKOCEvaluatorSIMD{S,T},
    A::AbstractMatrix{T},
) where {S,T}

    # Validate array dimensions.
    stage_axis, _, _ = get_axes(ev)
    @static if PERFORM_INTERNAL_BOUNDS_CHECKS
        @assert axes(A) == (stage_axis, stage_axis)
    end

    # Construct numeric constants.
    _zeros = zero(Vec{S,T})
    _ones = one(Vec{S,T})

    # Load columns of A into SIMD registers.
    A_cols = vload_cols(Val{S}(), A)

    # Iterate over Butcher instructions.
    @inbounds for (k, instruction) in pairs(ev.table.instructions)
        p, q = instruction.left, instruction.right
        if p == NULL_INDEX
            # The Butcher weight vector for the trivial tree
            # (p == NULL_INDEX and q == NULL_INDEX) is the all-ones vector.
            @assert q == NULL_INDEX
            ev.Phi[k] = _ones
        elseif q == NULL_INDEX
            # The Butcher weight vector for a tree obtained by extension
            # (p != NULL_INDEX and q == NULL_INDEX) is a matrix-vector product.
            phi = ev.Phi[p]
            result = _zeros
            for i = 1:S
                result += A_cols[i] * phi[i]
            end
            ev.Phi[k] = result
        else
            # The Butcher weight vector for a tree obtained by rooted sum
            # (p != NULL_INDEX and q != NULL_INDEX) is an elementwise product.
            ev.Phi[k] = ev.Phi[p] * ev.Phi[q]
        end
    end

    return ev
end


function compute_residual(
    ev::RKOCEvaluator{T},
    b::AbstractVector{T},
    i::Integer,
) where {T}

    # Validate array dimensions.
    stage_axis, _, output_axis = get_axes(ev)
    @static if PERFORM_INTERNAL_BOUNDS_CHECKS
        @assert axes(b) == (stage_axis,)
        @assert i in output_axis
    end

    return @inbounds begin
        # Compute dot product without SIMD for determinism.
        lhs = zero(T)
        k = ev.table.selected_indices[i]
        for j in stage_axis
            lhs += b[j] * ev.Phi[j, k]
        end
        # Subtract inv_gamma at the end for improved numerical stability.
        return lhs - ev.inv_gamma[i]
    end
end


function compute_residual(
    ev::RKOCEvaluatorSIMD{S,T},
    b::AbstractVector{T},
    i::Integer,
) where {S,T}

    # Validate array dimensions.
    stage_axis, _, output_axis = get_axes(ev)
    @static if PERFORM_INTERNAL_BOUNDS_CHECKS
        @assert axes(b) == (stage_axis,)
        @assert i in output_axis
    end

    return @inbounds begin
        phi = ev.Phi[ev.table.selected_indices[i]]
        vb = vload_vec(Val{S}(), b)
        # Subtract inv_gamma at the end for improved numerical stability.
        return vsum(vb * phi) - ev.inv_gamma[i]
    end
end


function compute_residuals!(
    residuals::AbstractVector{T},
    ev::RKOCEvaluator{T},
    b::AbstractVector{T},
) where {T}

    # Validate array dimensions.
    stage_axis, _, output_axis = get_axes(ev)
    @static if PERFORM_INTERNAL_BOUNDS_CHECKS
        @assert axes(residuals) == (output_axis,)
        @assert axes(b) == (stage_axis,)
    end

    # Compute residuals.
    @inbounds for i in output_axis
        residuals[i] = compute_residual(ev, b, i)
    end

    return residuals
end


function compute_residuals!(
    residuals::AbstractVector{T},
    ev::RKOCEvaluatorSIMD{S,T},
    b::AbstractVector{T},
) where {S,T}

    # Validate array dimensions.
    stage_axis, _, output_axis = get_axes(ev)
    @static if PERFORM_INTERNAL_BOUNDS_CHECKS
        @assert Base.require_one_based_indexing(residuals)
        @assert axes(residuals) == (output_axis,)
        @assert axes(b) == (stage_axis,)
    end

    # Construct numeric constants.
    _zeros = zero(Vec{S,T})

    # Load b into SIMD registers.
    vb = vload_vec(Val{S}(), b)

    # Obtain pointers to internal workspace arrays.
    residuals_ptr = pointer(residuals)
    selected_indices_ptr = pointer(ev.table.selected_indices)
    Phi_ptr = reinterpret(Ptr{T}, pointer(ev.Phi))
    inv_gamma_ptr = pointer(ev.inv_gamma)

    # Compute pointer increments.
    int_size = sizeof(Int)
    entry_size = sizeof(T)
    int_row_size = S * int_size
    row_size = S * entry_size

    @inbounds begin
        i = 0
        output_length = length(output_axis)

        # Compute residuals in SIMD batches.
        while i + S <= output_length
            indices = vload(Vec{S,Int}, selected_indices_ptr)
            Phi_row_ptrs = Phi_ptr + (indices - 1) * row_size
            lhs = _zeros
            for j = 1:S
                lhs += vgather(Phi_row_ptrs) * vb[j]
                Phi_row_ptrs += entry_size
            end
            # Subtract inv_gamma at the end for improved numerical stability.
            inv_gammas = vload(Vec{S,T}, inv_gamma_ptr)
            vstore(lhs - inv_gammas, residuals_ptr, nothing)
            # Advance pointers for next iteration.
            i += S
            residuals_ptr += row_size
            selected_indices_ptr += int_row_size
            inv_gamma_ptr += row_size
        end

        # Transition from 0-based to 1-based indexing.
        i += 1

        # Compute remaining residuals.
        while i <= output_length
            residuals[i] = compute_residual(ev, b, i)
            i += 1
        end
    end

    return residuals
end


############################## DIRECTIONAL DERIVATIVE COMPUTATION (FORWARD-MODE)


function pushforward_dPhi!(
    ev::RKOCEvaluator{T},
    A::AbstractMatrix{T},
    dA::AbstractMatrix{T},
) where {T}

    # Validate array dimensions.
    stage_axis, _, _ = get_axes(ev)
    @static if PERFORM_INTERNAL_BOUNDS_CHECKS
        @assert axes(A) == (stage_axis, stage_axis)
        @assert axes(dA) == (stage_axis, stage_axis)
    end

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
    @static if PERFORM_INTERNAL_BOUNDS_CHECKS
        @assert axes(dresiduals) == (output_axis,)
        @assert axes(b) == (stage_axis,)
        @assert axes(db) == (stage_axis,)
    end

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


################################## PARTIAL DERIVATIVE COMPUTATION (FORWARD-MODE)


function pushforward_dPhi!(
    ev::RKOCEvaluator{T},
    A::AbstractMatrix{T},
    u::Integer,
    v::Integer,
) where {T}

    # Validate array dimensions.
    stage_axis, _, _ = get_axes(ev)
    @static if PERFORM_INTERNAL_BOUNDS_CHECKS
        @assert axes(A) == (stage_axis, stage_axis)
        @assert u in stage_axis
        @assert v in stage_axis
    end

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
    @static if PERFORM_INTERNAL_BOUNDS_CHECKS
        @assert axes(dresiduals) == (output_axis,)
        @assert axes(b) == (stage_axis,)
    end

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


###################################################### COST FUNCTION DERIVATIVES


function (::RKCostL2{T})(
    adj::RKOCAdjoint{T,RKOCEvaluator{T}},
    b::AbstractVector{T},
) where {T}

    # Validate array dimensions.
    stage_axis, _, _ = get_axes(adj.ev)
    @static if PERFORM_INTERNAL_BOUNDS_CHECKS
        @assert axes(b) == (stage_axis,)
    end

    # Construct numeric constants.
    _zero = zero(T)

    # Initialize dPhi using derivative of L2 norm.
    @inbounds for (k, i) in Iterators.reverse(pairs(adj.ev.table.source_indices))
        if i == NULL_INDEX
            @simd ivdep for j in stage_axis
                adj.ev.dPhi[j, k] = _zero
            end
        else
            residual = compute_residual(adj.ev, b, i)
            derivative = residual + residual
            @simd ivdep for j in stage_axis
                adj.ev.dPhi[j, k] = derivative * b[j]
            end
        end
    end

    return adj
end


function (::RKCostL2{T})(
    adj::RKOCAdjoint{T,RKOCEvaluatorSIMD{S,T}},
    b::AbstractVector{T},
) where {S,T}

    # Validate array dimensions.
    stage_axis, _, _ = get_axes(adj.ev)
    @static if PERFORM_INTERNAL_BOUNDS_CHECKS
        @assert axes(b) == (stage_axis,)
    end

    # Construct numeric constants.
    _zeros = zero(Vec{S,T})

    # Load b into SIMD registers.
    vb = vload_vec(Val{S}(), b)

    # Initialize dPhi using derivative of L2 norm.
    @inbounds for (k, i) in Iterators.reverse(pairs(adj.ev.table.source_indices))
        if i == NULL_INDEX
            adj.ev.dPhi[k] = _zeros
        else
            residual = compute_residual(adj.ev, b, i)
            derivative = residual + residual
            adj.ev.dPhi[k] = derivative * vb
        end
    end

    return adj
end


function (::RKCostL2{T})(
    db::AbstractVector{T},
    adj::RKOCAdjoint{T,RKOCEvaluator{T}},
    b::AbstractVector{T},
) where {T}

    # Validate array dimensions.
    stage_axis, _, _ = get_axes(adj.ev)
    @static if PERFORM_INTERNAL_BOUNDS_CHECKS
        @assert axes(db) == (stage_axis,)
        @assert axes(b) == (stage_axis,)
    end

    # Construct numeric constants.
    _zero = zero(T)

    # Initialize db to zero.
    @simd ivdep for i in stage_axis
        @inbounds db[i] = _zero
    end

    # Compute db using derivative of L2 norm.
    @inbounds for (i, k) in pairs(adj.ev.table.selected_indices)
        residual = compute_residual(adj.ev, b, i)
        derivative = residual + residual
        @simd ivdep for j in stage_axis
            db[j] += derivative * adj.ev.Phi[j, k]
        end
    end

    return db
end


function (::RKCostL2{T})(
    db::AbstractVector{T},
    adj::RKOCAdjoint{T,RKOCEvaluatorSIMD{S,T}},
    b::AbstractVector{T},
) where {S,T}

    # Validate array dimensions.
    stage_axis, _, _ = get_axes(adj.ev)
    @static if PERFORM_INTERNAL_BOUNDS_CHECKS
        @assert axes(db) == (stage_axis,)
        @assert axes(b) == (stage_axis,)
    end

    # Initialize db to zero.
    vdb = zero(Vec{S,T})

    # Compute db using derivative of L2 norm.
    @inbounds for (i, k) in pairs(adj.ev.table.selected_indices)
        residual = compute_residual(adj.ev, b, i)
        derivative = residual + residual
        vdb += derivative * adj.ev.Phi[k]
    end

    vstore_vec!(db, vdb)
    return db
end


############################################ GRADIENT COMPUTATION (REVERSE-MODE)


function pullback_dPhi!(
    ev::RKOCEvaluator{T},
    A::AbstractMatrix{T},
) where {T}

    # Validate array dimensions.
    stage_axis, internal_axis, _ = get_axes(ev)
    @static if PERFORM_INTERNAL_BOUNDS_CHECKS
        @assert axes(A) == (stage_axis, stage_axis)
    end

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


function pullback_dPhi!(
    ev::RKOCEvaluatorSIMD{S,T},
    A::AbstractMatrix{T},
) where {S,T}

    # Validate array dimensions.
    stage_axis, internal_axis, _ = get_axes(ev)
    @static if PERFORM_INTERNAL_BOUNDS_CHECKS
        @assert axes(A) == (stage_axis, stage_axis)
    end

    # Load rows of A into SIMD registers.
    A_rows = vload_rows(Val{S}(), A)

    # Iterate over intermediate trees in reverse order.
    @inbounds for k in Iterators.reverse(internal_axis)
        c = ev.table.extension_indices[k]
        if c != NULL_INDEX
            # Perform adjoint matrix-vector multiplication.
            dphi = ev.dPhi[c]
            result = ev.dPhi[k]
            for i = 1:S
                result += A_rows[i] * dphi[i]
            end
            ev.dPhi[k] = result
        end
        for i in ev.table.rooted_sum_ranges[k]
            (p, q) = ev.table.rooted_sum_indices[i]
            ev.dPhi[k] += ev.Phi[p] * ev.dPhi[q]
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
    @static if PERFORM_INTERNAL_BOUNDS_CHECKS
        @assert axes(dA) == (stage_axis, stage_axis)
    end

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


function pullback_dA!(
    dA::AbstractMatrix{T},
    ev::RKOCEvaluatorSIMD{S,T},
) where {S,T}

    # Validate array dimensions.
    stage_axis, _, _ = get_axes(ev)
    @static if PERFORM_INTERNAL_BOUNDS_CHECKS
        @assert axes(dA) == (stage_axis, stage_axis)
    end

    # Construct numeric constants.
    _zeros = zero(Vec{S,T})

    # Initialize dA to zero.
    dA_ptr = pointer(dA)
    dA_row_ptr = dA_ptr
    row_size = S * sizeof(T)
    for _ = 1:S
        vstore(_zeros, dA_row_ptr, nothing)
        dA_row_ptr += row_size
    end

    # Iterate over intermediate trees obtained by extension.
    @inbounds for (k, c) in pairs(ev.table.extension_indices)
        if c != NULL_INDEX
            phi = ev.Phi[k]
            dA_row_ptr = dA_ptr
            for t = 1:S
                dA_row = vload(Vec{S,T}, dA_row_ptr)
                vstore(dA_row + phi[t] * ev.dPhi[c], dA_row_ptr, nothing)
                dA_row_ptr += row_size
            end
        end
    end

    return dA
end


################################################################################


export RKOCOptimizationProblem


struct RKOCOptimizationProblem{T,E,C,P}
    ev::E
    cost::C
    param::P
    A::Matrix{T}
    dA::Matrix{T}
    b::Vector{T}
    db::Vector{T}
end


function RKOCOptimizationProblem(ev::E, cost::C, param::P) where {T,
    E<:AbstractRKOCEvaluator{T},
    C<:AbstractRKCost{T},
    P<:AbstractRKParameterization{T}}

    stage_axis, _, _ = get_axes(ev)
    @assert param.num_stages == length(stage_axis)

    A = Matrix{T}(undef, param.num_stages, param.num_stages)
    dA = Matrix{T}(undef, param.num_stages, param.num_stages)
    b = Vector{T}(undef, param.num_stages)
    db = Vector{T}(undef, param.num_stages)
    return RKOCOptimizationProblem(ev, cost, param, A, dA, b, db)
end


struct RKOCOptimizationProblemAdjoint{T,E,C,P}
    prob::RKOCOptimizationProblem{T,E,C,P}
end


@inline Base.adjoint(prob::RKOCOptimizationProblem{T,E,C,P}) where {T,E,C,P} =
    RKOCOptimizationProblemAdjoint{T,E,C,P}(prob)


function (prob::RKOCOptimizationProblem{T,E,C,P})(
    x::AbstractVector{T},
) where {T,
    E<:AbstractRKOCEvaluator{T},
    C<:AbstractRKCost{T},
    P<:AbstractRKParameterization{T}}

    @assert length(x) == prob.param.num_variables

    prob.param(prob.A, prob.b, x)
    return prob.ev(prob.cost, prob.A, prob.b)
end


function (adj::RKOCOptimizationProblemAdjoint{T,E,C,P})(
    g::AbstractVector{T},
    x::AbstractVector{T},
) where {T,
    E<:AbstractRKOCEvaluator{T},
    C<:AbstractRKCost{T},
    P<:AbstractRKParameterization{T}}

    @assert axes(g) == axes(x)
    @assert length(g) == adj.prob.param.num_variables

    adj.prob.param(adj.prob.A, adj.prob.b, x)
    adj.prob.ev'(adj.prob.dA, adj.prob.db,
        adj.prob.cost, adj.prob.A, adj.prob.b)
    adj.prob.param(g, adj.prob.dA, adj.prob.db)
    return g
end


function (adj::RKOCOptimizationProblemAdjoint{T,E,C,P})(
    x::AbstractVector{T},
) where {T,
    E<:AbstractRKOCEvaluator{T},
    C<:AbstractRKCost{T},
    P<:AbstractRKParameterization{T}}

    @assert length(x) == adj.prob.param.num_variables

    return adj(similar(x), x)
end


end # module RungeKuttaToolKit
