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
using MultiFloats: MultiFloat, MultiFloatVec, mfvgather, mfvscatter, scale


####################################################### EVALUATOR DATA STRUCTURE


export RKOCEvaluator, RKOCEvaluatorSIMD, RKOCEvaluatorMFV, RKOCEvaluatorAO


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


struct RKOCEvaluatorMFV{S,T,N} <: AbstractRKOCEvaluator{MultiFloat{T,N}}
    table::ButcherInstructionTable
    Phi::Vector{MultiFloatVec{S,T,N}}
    dPhi::Vector{MultiFloatVec{S,T,N}}
    inv_gamma::Vector{MultiFloat{T,N}}
end


"""
    RKOCEvaluatorMFV{S,T,N}(
        trees::AbstractVector{LevelSequence},
    ) -> RKOCEvaluatorMFV{S,T,N}

Construct an `RKOCEvaluatorMFV` that encodes a given list of rooted trees.

Unlike a standard `RKOCEvaluator`, the number of stages `S` is provided as a
static type parameter in order to statically determine the SIMD vector width.

# Arguments
- `trees`: input list of rooted trees in `LevelSequence` representation.

If the type `T` is not specified, it defaults to `Float64`.
"""
function RKOCEvaluatorMFV{S,T,N}(
    trees::AbstractVector{LevelSequence},
) where {S,T,N}
    table = ButcherInstructionTable(trees)
    return RKOCEvaluatorMFV{S,T,N}(table,
        Vector{MultiFloatVec{S,T,N}}(undef, length(table.instructions)),
        Vector{MultiFloatVec{S,T,N}}(undef, length(table.instructions)),
        [inv(MultiFloat{T,N}(butcher_density(tree))) for tree in trees])
end


RKOCEvaluatorMFV{S,N}(trees::AbstractVector{LevelSequence}) where {S,N} =
    RKOCEvaluatorMFV{S,Float64,N}(trees)


"""
    RKOCEvaluatorMFV{S,T,N}(order::Integer) -> RKOCEvaluatorMFV{S,T,N}

Construct an `RKOCEvaluatorMFV` that encodes all rooted trees
having at most `order` vertices. This is equivalent to
`RKOCEvaluatorMFV{S,T}(all_rooted_trees(order))`.

Unlike a standard `RKOCEvaluator`, the number of stages `S` is provided as a
static type parameter in order to statically determine the SIMD vector width.

If the type `T` is not specified, it defaults to `Float64`.
"""
RKOCEvaluatorMFV{S,T,N}(order::Integer) where {S,T,N} =
    RKOCEvaluatorMFV{S,T,N}(all_rooted_trees(order))


RKOCEvaluatorMFV{S,N}(order::Integer) where {S,N} =
    RKOCEvaluatorMFV{S,Float64,N}(order)


######################################################### ADJOINT DATA STRUCTURE


struct RKOCAdjoint{T,E} <: AbstractRKOCAdjoint{T}
    ev::E
end


@inline Base.adjoint(ev::E) where {T,E<:AbstractRKOCEvaluator{T}} =
    RKOCAdjoint{T,E}(ev)


################################################################################


@inline function get_axes(ev::RKOCEvaluator)
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


@inline function get_axes(ev::RKOCEvaluatorSIMD{S}) where {S}
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


@inline function get_axes(ev::RKOCEvaluatorMFV{S}) where {S}
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


@inline function mfvload_vec(
    ::Val{M},
    v::Vector{MultiFloat{T,N}},
) where {M,T,N}
    @static if PERFORM_INTERNAL_BOUNDS_CHECKS
        @assert size(v) == (M,)
    end
    v_ptr = pointer(v)
    iota = Vec{M,Int}(ntuple(i -> (i - 1), Val{M}()))
    return mfvgather(v_ptr, iota)
end


@inline function vstore_vec!(v::Vector{T}, data::Vec{N,T}) where {N,T}
    @static if PERFORM_INTERNAL_BOUNDS_CHECKS
        @assert size(v) == (N,)
    end
    ptr = pointer(v)
    vstore(data, ptr, nothing)
    return v
end


@inline function mfvstore_vec!(
    v::Vector{MultiFloat{T,N}},
    data::MultiFloatVec{M,T,N},
) where {M,T,N}
    @static if PERFORM_INTERNAL_BOUNDS_CHECKS
        @assert size(v) == (M,)
    end
    v_ptr = pointer(v)
    iota = Vec{M,Int}(ntuple(i -> (i - 1), Val{M}()))
    mfvscatter(data, v_ptr, iota)
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


@inline function mfvload_cols(
    ::Val{M},
    A::Matrix{MultiFloat{T,N}},
) where {M,T,N}
    @static if PERFORM_INTERNAL_BOUNDS_CHECKS
        @assert size(A) == (M, M)
    end
    A_ptr = pointer(A)
    iota = Vec{M,Int}(ntuple(i -> (i - 1), Val{M}()))
    return ntuple(i -> mfvgather(A_ptr, iota + (i - 1) * M), Val{M}())
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


@inline function mfvload_rows(
    ::Val{M},
    A::Matrix{MultiFloat{T,N}},
) where {M,T,N}
    @static if PERFORM_INTERNAL_BOUNDS_CHECKS
        @assert size(A) == (M, M)
    end
    A_ptr = pointer(A)
    iota = Vec{M,Int}(ntuple(i -> (i - 1) * M, Val{M}()))
    return ntuple(i -> mfvgather(A_ptr, iota + (i - 1)), Val{M}())
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


@inline function mfvsum(v::MultiFloatVec{M,T,N}) where {M,T,N}
    # Compute dot products sequentially for determinism.
    # SIMD parallel sum instructions can change summation order.
    result = zero(MultiFloat{T,N})
    for i = 1:M
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


function compute_Phi!(
    ev::RKOCEvaluatorMFV{S,T,N},
    A::AbstractMatrix{MultiFloat{T,N}},
) where {S,T,N}

    # Validate array dimensions.
    stage_axis, _, _ = get_axes(ev)
    @static if PERFORM_INTERNAL_BOUNDS_CHECKS
        @assert axes(A) == (stage_axis, stage_axis)
    end

    # Construct numeric constants.
    _zeros = zero(MultiFloatVec{S,T,N})
    _ones = one(MultiFloatVec{S,T,N})

    # Load columns of A into SIMD registers.
    A_cols = mfvload_cols(Val{S}(), A)

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


function compute_residual(
    ev::RKOCEvaluatorMFV{S,T,N},
    b::AbstractVector{MultiFloat{T,N}},
    i::Integer,
) where {S,T,N}

    # Validate array dimensions.
    stage_axis, _, output_axis = get_axes(ev)
    @static if PERFORM_INTERNAL_BOUNDS_CHECKS
        @assert axes(b) == (stage_axis,)
        @assert i in output_axis
    end

    return @inbounds begin
        phi = ev.Phi[ev.table.selected_indices[i]]
        vb = mfvload_vec(Val{S}(), b)
        # Subtract inv_gamma at the end for improved numerical stability.
        return mfvsum(vb * phi) - ev.inv_gamma[i]
    end
end


function compute_residuals!(
    residuals::AbstractVector{T},
    ev::AbstractRKOCEvaluator{T},
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


#=

This is an attempt at a SIMD-optimized implementation of `compute_residuals!`,
but it does not work when the SIMD vector length `S` is not a power of two.

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

=#


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


function pullback_dPhi!(
    ev::RKOCEvaluatorMFV{S,T,N},
    A::AbstractMatrix{MultiFloat{T,N}},
) where {S,T,N}

    # Validate array dimensions.
    stage_axis, internal_axis, _ = get_axes(ev)
    @static if PERFORM_INTERNAL_BOUNDS_CHECKS
        @assert axes(A) == (stage_axis, stage_axis)
    end

    # Load rows of A into SIMD registers.
    A_rows = mfvload_rows(Val{S}(), A)

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
    dA_col_ptr = dA_ptr
    col_size = S * sizeof(T)
    for _ = 1:S
        vstore(_zeros, dA_col_ptr, nothing)
        dA_col_ptr += col_size
    end

    # Iterate over intermediate trees obtained by extension.
    @inbounds for (k, c) in pairs(ev.table.extension_indices)
        if c != NULL_INDEX
            phi = ev.Phi[k]
            dA_col_ptr = dA_ptr
            for t = 1:S
                dA_row = vload(Vec{S,T}, dA_col_ptr)
                vstore(dA_row + phi[t] * ev.dPhi[c], dA_col_ptr, nothing)
                dA_col_ptr += col_size
            end
        end
    end

    return dA
end


function pullback_dA!(
    dA::AbstractMatrix{MultiFloat{T,N}},
    ev::RKOCEvaluatorMFV{S,T,N},
) where {S,T,N}

    # Validate array dimensions.
    stage_axis, _, _ = get_axes(ev)
    @static if PERFORM_INTERNAL_BOUNDS_CHECKS
        @assert axes(dA) == (stage_axis, stage_axis)
    end

    # Construct numeric constants.
    _zeros = zero(MultiFloatVec{S,T,N})

    # Initialize dA to zero.
    dA_ptr = pointer(dA)
    iota = Vec{S,Int}(ntuple(i -> (i - 1), Val{S}()))
    dA_col_ptr = dA_ptr
    col_size = S * sizeof(T) * N
    for _ = 1:S
        mfvscatter(_zeros, dA_col_ptr, iota)
        dA_col_ptr += col_size
    end

    # Iterate over intermediate trees obtained by extension.
    @inbounds for (k, c) in pairs(ev.table.extension_indices)
        if c != NULL_INDEX
            phi = ev.Phi[k]
            dA_col_ptr = dA_ptr
            for t = 1:S
                dA_row = mfvgather(dA_col_ptr, iota)
                mfvscatter(dA_row + phi[t] * ev.dPhi[c], dA_col_ptr, iota)
                dA_col_ptr += col_size
            end
        end
    end

    return dA
end


################################################################################


include("L2Gradients.jl")


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


################################################################################


export ConstrainedRKOCOptimizer


using LinearAlgebra: axpy!, axpby!, dot, mul!


struct ConstrainedRKOCOptimizer{T,E,P}
    ev::E
    param::P
    A::Matrix{T}
    b::Vector{T}
    x::Vector{T}
    lambda::Vector{T}
    joint_residual::Vector{T}
    constraint_residual::SubArray{T,1,Vector{T},Tuple{UnitRange{Int}},true}
    objective_residual::SubArray{T,1,Vector{T},Tuple{UnitRange{Int}},true}
    joint_jacobian::Matrix{T}
    constraint_jacobian::SubArray{T,2,Matrix{T},Tuple{UnitRange{Int},UnitRange{Int}},false}
    objective_jacobian::SubArray{T,2,Matrix{T},Tuple{UnitRange{Int},UnitRange{Int}},false}
    frontier::Vector{Tuple{T,T}}
    x_trial::Vector{T}
    kkt_temp::Vector{T}
    kkt_delta::Vector{T}
    kkt_delta_x::SubArray{T,1,Vector{T},Tuple{UnitRange{Int}},true}
    kkt_delta_lambda::SubArray{T,1,Vector{T},Tuple{UnitRange{Int}},true}
    kkt_residual::Vector{T}
    kkt_residual_x::SubArray{T,1,Vector{T},Tuple{UnitRange{Int}},true}
    kkt_residual_lambda::SubArray{T,1,Vector{T},Tuple{UnitRange{Int}},true}
    step_direction::Vector{T}
    step_direction_x::SubArray{T,1,Vector{T},Tuple{UnitRange{Int}},true}
    step_direction_lambda::SubArray{T,1,Vector{T},Tuple{UnitRange{Int}},true}
    conjugate_step_direction::Vector{T}
    conjugate_step_direction_x::SubArray{T,1,Vector{T},Tuple{UnitRange{Int}},true}
    conjugate_step_direction_lambda::SubArray{T,1,Vector{T},Tuple{UnitRange{Int}},true}
end


function ConstrainedRKOCOptimizer(
    constraint_trees::AbstractVector{LevelSequence},
    objective_trees::AbstractVector{LevelSequence},
    param::P,
    x::AbstractVector{T},
) where {T,P<:AbstractRKParameterization{T}}

    @assert length(x) == param.num_variables

    nc = length(constraint_trees)
    no = length(objective_trees)
    ns = param.num_stages
    nv = length(x)

    A = Matrix{T}(undef, ns, ns)
    b = Vector{T}(undef, ns)
    x = copy(x)
    param(A, b, x)

    ev = RKOCEvaluator{T}(vcat(constraint_trees, objective_trees), ns)

    joint_residual = Vector{T}(undef, nc + no)
    ev(joint_residual, A, b)
    constraint_residual = @view joint_residual[1:nc]
    objective_residual = @view joint_residual[nc+1:nc+no]

    joint_jacobian = Matrix{T}(undef, nc + no, nv)
    param(joint_jacobian, A, b, ev, x)
    constraint_jacobian = @view joint_jacobian[1:nc, 1:nv]
    objective_jacobian = @view joint_jacobian[nc+1:nc+no, 1:nv]

    frontier = [(
        sum(abs2, @view joint_residual[1:nc]),
        sum(abs2, @view joint_residual[nc+1:nc+no]))]

    x_trial = Vector{T}(undef, nv)
    kkt_temp = Vector{T}(undef, no)
    kkt_delta = Vector{T}(undef, nv + nc)
    kkt_delta_x = @view kkt_delta[1:nv]
    kkt_delta_lambda = @view kkt_delta[nv+1:end]
    kkt_residual = Vector{T}(undef, nv + nc)
    kkt_residual_x = @view kkt_residual[1:nv]
    kkt_residual_lambda = @view kkt_residual[nv+1:end]
    step_direction = Vector{T}(undef, nv + nc)
    step_direction_x = @view step_direction[1:nv]
    step_direction_lambda = @view step_direction[nv+1:end]
    conjugate_step_direction = Vector{T}(undef, nv + nc)
    conjugate_step_direction_x = @view conjugate_step_direction[1:nv]
    conjugate_step_direction_lambda = @view conjugate_step_direction[nv+1:end]

    return ConstrainedRKOCOptimizer{T,RKOCEvaluator{T},P}(
        ev, param, A, b, x, zeros(T, nc),
        joint_residual, constraint_residual, objective_residual,
        joint_jacobian, constraint_jacobian, objective_jacobian,
        frontier, x_trial, kkt_temp,
        kkt_delta, kkt_delta_x, kkt_delta_lambda,
        kkt_residual, kkt_residual_x, kkt_residual_lambda,
        step_direction, step_direction_x, step_direction_lambda,
        conjugate_step_direction, conjugate_step_direction_x,
        conjugate_step_direction_lambda)
end


function _kkt_mul!(
    opt::ConstrainedRKOCOptimizer{T,E,P},
    w_x::AbstractVector{T},
    w_lambda::AbstractVector{T},
    v_x::AbstractVector{T},
    v_lambda::AbstractVector{T},
) where {T,E,P}
    _one = one(T)
    mul!(opt.kkt_temp, opt.objective_jacobian, v_x)
    mul!(w_x, opt.objective_jacobian', opt.kkt_temp)
    mul!(w_x, opt.constraint_jacobian', v_lambda, _one, _one)
    mul!(w_lambda, opt.constraint_jacobian, v_x)
    return opt
end


function _kkt_solve!(opt::ConstrainedRKOCOptimizer{T,E,P}) where {T,E,P}
    _zero = zero(T)
    _one = one(T)

    opt.kkt_delta .= _zero
    mul!(opt.kkt_residual_x, opt.objective_jacobian', opt.objective_residual)
    mul!(opt.kkt_residual_x, opt.constraint_jacobian', opt.lambda, _one, _one)
    copy!(opt.kkt_residual_lambda, opt.constraint_residual)
    copy!(opt.step_direction, opt.kkt_residual)
    r_norm2 = dot(opt.kkt_residual, opt.kkt_residual)

    while true
        _kkt_mul!(opt, opt.conjugate_step_direction_x,
            opt.conjugate_step_direction_lambda,
            opt.step_direction_x, opt.step_direction_lambda)
        step_size = r_norm2 / dot(opt.step_direction,
            opt.conjugate_step_direction)
        axpy!(step_size, opt.step_direction, opt.kkt_delta)
        axpy!(-step_size, opt.conjugate_step_direction, opt.kkt_residual)
        r_norm2_new = dot(opt.kkt_residual, opt.kkt_residual)
        if !(r_norm2_new < r_norm2)
            break
        end
        axpby!(_one, opt.kkt_residual,
            r_norm2_new / r_norm2, opt.step_direction)
        r_norm2 = r_norm2_new
    end

    return opt
end


@inline _dominates(a::Tuple{T,T}, b::Tuple{T,T}) where {T} =
    (a[1] <= b[1]) & (a[2] <= b[2])


function _update_frontier!(
    score::Tuple{T,T},
    frontier::AbstractVector{Tuple{T,T}},
) where {T}
    if !any(_dominates(point, score) for point in frontier)
        filter!(point -> !_dominates(score, point), frontier)
        push!(frontier, score)
        return true
    else
        return false
    end
end


_mfbase(::Type{T}) where {T} = T
_mfbase(::Type{MultiFloat{T,N}}) where {T,N} = T
_mfbase(::Type{MultiFloatVec{M,T,N}}) where {M,T,N} = T


function (opt::ConstrainedRKOCOptimizer{T,E,P})() where {T,E,P}

    _one = one(_mfbase(T))
    _two = _one + _one
    _half = inv(_two)

    _kkt_solve!(opt)

    alpha = _one
    while true
        @simd ivdep for i in eachindex(opt.x_trial)
            @inbounds opt.x_trial[i] = opt.x[i] - scale(alpha, opt.kkt_delta_x[i])
        end
        if any(isnan, opt.x_trial) || (opt.x_trial == opt.x)
            return false
        end
        opt.param(opt.A, opt.b, opt.x_trial)
        opt.ev(opt.joint_residual, opt.A, opt.b)
        score = (
            sum(abs2, opt.constraint_residual),
            sum(abs2, opt.objective_residual))
        if _update_frontier!(score, opt.frontier)
            copy!(opt.x, opt.x_trial)
            opt.param(opt.joint_jacobian, opt.A, opt.b, opt.ev, opt.x)
            @simd ivdep for i in eachindex(opt.lambda)
                @inbounds opt.lambda[i] -= scale(alpha, opt.kkt_block_2[i])
            end
            return true
        else
            alpha *= _half
        end
    end
end


end # module RungeKuttaToolKit
