module RKCost


using ..RungeKuttaToolKit: PERFORM_INTERNAL_BOUNDS_CHECKS, NULL_INDEX,
    AbstractRKOCEvaluator, AbstractRKOCAdjoint, AbstractRKCost,
    get_axes, compute_residual


############################################################################# L1


export RKCostL1


struct RKCostL1{T} <: AbstractRKCost{T} end


function (::RKCostL1{T})(
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


export RKCostWeightedL1


struct RKCostWeightedL1{T} <: AbstractRKCost{T}
    weights::Vector{T}
end


function (obj::RKCostWeightedL1{T})(
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


export RKCostL2


struct RKCostL2{T} <: AbstractRKCost{T} end


function (::RKCostL2{T})(
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


function (::RKCostL2{T})(
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


function (::RKCostL2{T})(
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


export RKCostWeightedL2


struct RKCostWeightedL2{T} <: AbstractRKCost{T}
    weights::Vector{T}
end


function (obj::RKCostWeightedL2{T})(
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


function (obj::RKCostWeightedL2{T})(
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


function (obj::RKCostWeightedL2{T})(
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


export RKCostLInfinity


struct RKCostLInfinity{T} <: AbstractRKCost{T} end


function (::RKCostLInfinity{T})(
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


export RKCostWeightedLInfinity


struct RKCostWeightedLInfinity{T} <: AbstractRKCost{T}
    weights::Vector{T}
end


function (obj::RKCostWeightedLInfinity{T})(
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


export RKCostHuber


struct RKCostHuber{T} <: AbstractRKCost{T}
    delta::T
end


function huber_loss(delta::T, x::T) where {T}
    abs_x = abs(x)
    return ((abs_x <= delta) ? (abs_x * abs_x) :
            (delta * ((abs_x + abs_x) - delta)))
end


function (obj::RKCostHuber{T})(
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


export RKCostWeightedHuber


struct RKCostWeightedHuber{T} <: AbstractRKCost{T}
    delta::T
    weights::Vector{T}
end


function (obj::RKCostWeightedHuber{T})(
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


end # module RKCost
