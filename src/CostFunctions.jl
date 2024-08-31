module CostFunctions


using ..RungeKuttaToolKit: PERFORM_INTERNAL_BOUNDS_CHECKS, NULL_INDEX,
    AbstractRKOCEvaluator, AbstractRKOCAdjoint, AbstractRKCost,
    get_axes, compute_residual


############################################################################# L1


export L1RKCost


struct L1RKCost{T} <: AbstractRKCost{T} end


function (::L1RKCost{T})(
    ev::AbstractRKOCEvaluator{T},
    b::AbstractVector{T},
) where {T}

    # Validate array dimensions.
    stage_axis, _, output_axis = get_axes(ev)
    @static if PERFORM_INTERNAL_BOUNDS_CHECKS
        @assert axes(b) == (stage_axis,)
    end

    # Compute L1 norm of residuals.
    result = zero(T)
    @inbounds for i in output_axis
        result += abs(compute_residual(ev, b, i))
    end

    return result
end


#################################################################### WEIGHTED L1


export WeightedL1RKCost


struct WeightedL1RKCost{T} <: AbstractRKCost{T}
    weights::Vector{T}
end


function (obj::WeightedL1RKCost{T})(
    ev::AbstractRKOCEvaluator{T},
    b::AbstractVector{T},
) where {T}

    # Validate array dimensions.
    stage_axis, _, output_axis = get_axes(ev)
    @static if PERFORM_INTERNAL_BOUNDS_CHECKS
        @assert axes(obj.weights) == (output_axis,)
        @assert axes(b) == (stage_axis,)
    end

    # Compute weighted L1 norm of residuals.
    result = zero(T)
    @inbounds for i in output_axis
        result += obj.weights[i] * abs(compute_residual(ev, b, i))
    end

    return result
end


############################################################################# L2


export L2RKCost


struct L2RKCost{T} <: AbstractRKCost{T} end


function (::L2RKCost{T})(
    ev::AbstractRKOCEvaluator{T},
    b::AbstractVector{T},
) where {T}

    # Validate array dimensions.
    stage_axis, _, output_axis = get_axes(ev)
    @static if PERFORM_INTERNAL_BOUNDS_CHECKS
        @assert axes(b) == (stage_axis,)
    end

    # Compute L2 norm of residuals.
    result = zero(T)
    @inbounds for i in output_axis
        result += abs2(compute_residual(ev, b, i))
    end

    return result
end


function (::L2RKCost{T})(
    adj::AbstractRKOCAdjoint{T},
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


function (::L2RKCost{T})(
    db::AbstractVector{T},
    adj::AbstractRKOCAdjoint{T},
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


#################################################################### WEIGHTED L2


export WeightedL2RKCost


struct WeightedL2RKCost{T} <: AbstractRKCost{T}
    weights::Vector{T}
end


function (obj::WeightedL2RKCost{T})(
    ev::AbstractRKOCEvaluator{T},
    b::AbstractVector{T},
) where {T}

    # Validate array dimensions.
    stage_axis, _, output_axis = get_axes(ev)
    @static if PERFORM_INTERNAL_BOUNDS_CHECKS
        @assert axes(obj.weights) == (output_axis,)
        @assert axes(b) == (stage_axis,)
    end

    # Compute weighted L2 norm of residuals.
    result = zero(T)
    @inbounds for i in output_axis
        result += obj.weights[i] * abs2(compute_residual(ev, b, i))
    end

    return result
end


function (obj::WeightedL2RKCost{T})(
    adj::AbstractRKOCAdjoint{T},
    b::AbstractVector{T},
) where {T}

    # Validate array dimensions.
    stage_axis, _, output_axis = get_axes(adj.ev)
    @static if PERFORM_INTERNAL_BOUNDS_CHECKS
        @assert axes(obj.weights) == (output_axis,)
        @assert axes(b) == (stage_axis,)
    end

    # Construct numeric constants.
    _zero = zero(T)

    # Initialize dPhi using derivative of weighted L2 norm.
    @inbounds for (k, i) in Iterators.reverse(pairs(adj.ev.table.source_indices))
        if i == NULL_INDEX
            @simd ivdep for j in stage_axis
                adj.ev.dPhi[j, k] = _zero
            end
        else
            residual = compute_residual(adj.ev, b, i)
            derivative = obj.weights[i] * (residual + residual)
            @simd ivdep for j in stage_axis
                adj.ev.dPhi[j, k] = derivative * b[j]
            end
        end
    end

    return adj
end


function (obj::WeightedL2RKCost{T})(
    db::AbstractVector{T},
    adj::AbstractRKOCAdjoint{T},
    b::AbstractVector{T},
) where {T}

    # Validate array dimensions.
    stage_axis, _, output_axis = get_axes(adj.ev)
    @static if PERFORM_INTERNAL_BOUNDS_CHECKS
        @assert axes(obj.weights) == (output_axis,)
        @assert axes(db) == (stage_axis,)
        @assert axes(b) == (stage_axis,)
    end

    # Construct numeric constants.
    _zero = zero(T)

    # Initialize db to zero.
    @simd ivdep for i in stage_axis
        @inbounds db[i] = _zero
    end

    # Compute db using derivative of weighted L2 norm.
    @inbounds for (i, k) in pairs(adj.ev.table.selected_indices)
        residual = compute_residual(adj.ev, b, i)
        derivative = obj.weights[i] * (residual + residual)
        @simd ivdep for j in stage_axis
            db[j] += derivative * adj.ev.Phi[j, k]
        end
    end

    return db
end


##################################################################### L-INFINITY


export LInfinityRKCost


struct LInfinityRKCost{T} <: AbstractRKCost{T} end


function (::LInfinityRKCost{T})(
    ev::AbstractRKOCEvaluator{T},
    b::AbstractVector{T},
) where {T}

    # Validate array dimensions.
    stage_axis, _, output_axis = get_axes(ev)
    @static if PERFORM_INTERNAL_BOUNDS_CHECKS
        @assert axes(b) == (stage_axis,)
    end

    # Compute LInfinity norm of residuals.
    result = zero(T)
    @inbounds for i in output_axis
        result = max(result, abs(compute_residual(ev, b, i)))
    end

    return result
end


############################################################ WEIGHTED L-INFINITY


export WeightedLInfinityRKCost


struct WeightedLInfinityRKCost{T} <: AbstractRKCost{T}
    weights::Vector{T}
end


function (obj::WeightedLInfinityRKCost{T})(
    ev::AbstractRKOCEvaluator{T},
    b::AbstractVector{T},
) where {T}

    # Validate array dimensions.
    stage_axis, _, output_axis = get_axes(ev)
    @static if PERFORM_INTERNAL_BOUNDS_CHECKS
        @assert axes(obj.weights) == (output_axis,)
        @assert axes(b) == (stage_axis,)
    end

    # Compute weighted LInfinity norm of residuals.
    result = zero(T)
    @inbounds for i in output_axis
        result = max(result, obj.weights[i] * abs(compute_residual(ev, b, i)))
    end

    return result
end


########################################################################## HUBER


export HuberRKCost


struct HuberRKCost{T} <: AbstractRKCost{T}
    delta::T
end


function huber_loss(delta::T, x::T) where {T}
    abs_x = abs(x)
    return ((abs_x <= delta) ? (abs_x * abs_x) :
            (delta * ((abs_x + abs_x) - delta)))
end


function (obj::HuberRKCost{T})(
    ev::AbstractRKOCEvaluator{T},
    b::AbstractVector{T}
) where {T}

    # Validate array dimensions.
    stage_axis, _, output_axis = get_axes(ev)
    @static if PERFORM_INTERNAL_BOUNDS_CHECKS
        @assert axes(b) == (stage_axis,)
    end

    # Compute sum of Huber losses of residuals.
    result = zero(T)
    @inbounds for i in output_axis
        result += huber_loss(obj.delta, compute_residual(ev, b, i))
    end

    return result
end


################################################################# WEIGHTED HUBER


export WeightedHuberRKCost


struct WeightedHuberRKCost{T} <: AbstractRKCost{T}
    delta::T
    weights::Vector{T}
end


function (obj::WeightedHuberRKCost{T})(
    ev::AbstractRKOCEvaluator{T},
    b::AbstractVector{T}
) where {T}

    # Validate array dimensions.
    stage_axis, _, output_axis = get_axes(ev)
    @static if PERFORM_INTERNAL_BOUNDS_CHECKS
        @assert axes(obj.weights) == (output_axis,)
        @assert axes(b) == (stage_axis,)
    end

    # Compute weighted sum of Huber losses of residuals.
    result = zero(T)
    @inbounds for i in output_axis
        result += obj.weights[i] * huber_loss(obj.delta,
            compute_residual(ev, b, i))
    end

    return result
end


end # module CostFunctions
