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


####################################################### EVALUATOR DATA STRUCTURE


export RKOCEvaluator, RKOCEvaluatorAO


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


struct RKOCAdjoint{T} <: AbstractRKOCAdjoint{T}
    ev::RKOCEvaluator{T}
end


@inline Base.adjoint(ev::RKOCEvaluator{T}) where {T} = RKOCAdjoint{T}(ev)


################################################################################


function get_axes(ev::RKOCEvaluator{T}) where {T}
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

    # Compute dot product without SIMD for determinism.
    lhs = zero(T)
    @inbounds begin
        k = ev.table.selected_indices[i]
        for j in stage_axis
            lhs += b[j] * ev.Phi[j, k]
        end
    end

    # Subtract inv_gamma at the end for improved numerical stability.
    return lhs - ev.inv_gamma[i]
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
