module RKCost


using ..RungeKuttaToolKit: NULL_INDEX, PERFORM_INTERNAL_BOUNDS_CHECKS,
    AbstractRKOCEvaluator, AbstractRKOCEvaluatorAO,
    AbstractRKOCAdjoint, AbstractRKOCAdjointAO, AbstractRKCost,
    get_axes, compute_residual


export AbstractRKCost


############################################################################# L1


export RKCostL1


"""
```math
\\sum_{t \\in T} \\left\\lvert
\\mathbf{b} \\cdot \\Phi_t(A) - \\frac{1}{\\gamma(t)}
\\right\\rvert
```
"""
struct RKCostL1{T} <: AbstractRKCost{T} end


@inline RKCostL1() = RKCostL1{Float64}()


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


function (::RKCostL1{T})(
    adj::AbstractRKOCAdjoint{T},
    b::AbstractVector{T},
) where {T}

    # Validate array dimensions.
    stage_axis, _, output_axis = get_axes(adj.ev)
    @static if PERFORM_INTERNAL_BOUNDS_CHECKS
        @assert axes(b) == (stage_axis,)
    end

    # Construct numeric constants.
    _zero = zero(T)

    # Compute tolerance for comparison to zero.
    max_abs_residual = _zero
    for i in output_axis
        max_abs_residual = max(max_abs_residual,
            abs(compute_residual(adj.ev, b, i)))
    end
    pos_tolerance = eps(T) * max_abs_residual
    neg_tolerance = -pos_tolerance

    # Initialize dPhi using derivative of L1 norm.
    @inbounds for (k, i) in Iterators.reverse(pairs(adj.ev.table.source_indices))
        if i == NULL_INDEX
            @simd ivdep for j in stage_axis
                adj.ev.dPhi[j, k] = _zero
            end
        else
            residual = compute_residual(adj.ev, b, i)
            if residual < neg_tolerance
                @simd ivdep for j in stage_axis
                    adj.ev.dPhi[j, k] = -b[j]
                end
            elseif residual > pos_tolerance
                @simd ivdep for j in stage_axis
                    adj.ev.dPhi[j, k] = b[j]
                end
            else
                @simd ivdep for j in stage_axis
                    adj.ev.dPhi[j, k] = _zero
                end
            end
        end
    end

    return adj
end


function (::RKCostL1{T})(
    db::AbstractVector{T},
    adj::AbstractRKOCAdjoint{T},
    b::AbstractVector{T},
) where {T}

    # Validate array dimensions.
    stage_axis, _, output_axis = get_axes(adj.ev)
    @static if PERFORM_INTERNAL_BOUNDS_CHECKS
        @assert axes(db) == (stage_axis,)
        @assert axes(b) == (stage_axis,)
    end

    # Construct numeric constants.
    _zero = zero(T)

    # Compute tolerance for comparison to zero.
    max_abs_residual = _zero
    for i in output_axis
        max_abs_residual = max(max_abs_residual,
            abs(compute_residual(adj.ev, b, i)))
    end
    pos_tolerance = eps(T) * max_abs_residual
    neg_tolerance = -pos_tolerance

    # Initialize db to zero.
    @simd ivdep for i in stage_axis
        @inbounds db[i] = _zero
    end

    # Compute db using derivative of L1 norm.
    @inbounds for (i, k) in pairs(adj.ev.table.selected_indices)
        residual = compute_residual(adj.ev, b, i)
        if residual < neg_tolerance
            @simd ivdep for j in stage_axis
                db[j] -= adj.ev.Phi[j, k]
            end
        elseif residual > pos_tolerance
            @simd ivdep for j in stage_axis
                db[j] += adj.ev.Phi[j, k]
            end
        end
    end

    return db
end


#################################################################### WEIGHTED L1


export RKCostWeightedL1


"""
```math
\\sum_{t \\in T} w_t \\left\\lvert
\\mathbf{b} \\cdot \\Phi_t(A) - \\frac{1}{\\gamma(t)}
\\right\\rvert
```
"""
struct RKCostWeightedL1{T} <: AbstractRKCost{T}
    weights::Vector{T}
end


function (cost::RKCostWeightedL1{T})(
    ev::AbstractRKOCEvaluator{T},
    b::AbstractVector{T},
) where {T}

    # Validate array dimensions.
    stage_axis, _, output_axis = get_axes(ev)
    @static if PERFORM_INTERNAL_BOUNDS_CHECKS
        @assert axes(cost.weights) == (output_axis,)
        @assert axes(b) == (stage_axis,)
    end

    # Compute weighted L1 norm of residuals.
    result = zero(T)
    @inbounds for i in output_axis
        result += cost.weights[i] * abs(compute_residual(ev, b, i))
    end

    return result
end


function (cost::RKCostWeightedL1{T})(
    adj::AbstractRKOCAdjoint{T},
    b::AbstractVector{T},
) where {T}

    # Validate array dimensions.
    stage_axis, _, output_axis = get_axes(adj.ev)
    @static if PERFORM_INTERNAL_BOUNDS_CHECKS
        @assert axes(b) == (stage_axis,)
    end

    # Construct numeric constants.
    _zero = zero(T)

    # Compute tolerance for comparison to zero.
    max_abs_residual = _zero
    for i in output_axis
        max_abs_residual = max(max_abs_residual,
            abs(compute_residual(adj.ev, b, i)))
    end
    pos_tolerance = eps(T) * max_abs_residual
    neg_tolerance = -pos_tolerance

    # Initialize dPhi using derivative of weighted L1 norm.
    @inbounds for (k, i) in Iterators.reverse(pairs(adj.ev.table.source_indices))
        if i == NULL_INDEX
            @simd ivdep for j in stage_axis
                adj.ev.dPhi[j, k] = _zero
            end
        else
            residual = compute_residual(adj.ev, b, i)
            if residual < neg_tolerance
                w = -cost.weights[i]
                @simd ivdep for j in stage_axis
                    adj.ev.dPhi[j, k] = w * b[j]
                end
            elseif residual > pos_tolerance
                w = cost.weights[i]
                @simd ivdep for j in stage_axis
                    adj.ev.dPhi[j, k] = w * b[j]
                end
            else
                @simd ivdep for j in stage_axis
                    adj.ev.dPhi[j, k] = _zero
                end
            end
        end
    end

    return adj
end


function (cost::RKCostWeightedL1{T})(
    db::AbstractVector{T},
    adj::AbstractRKOCAdjoint{T},
    b::AbstractVector{T},
) where {T}

    # Validate array dimensions.
    stage_axis, _, output_axis = get_axes(adj.ev)
    @static if PERFORM_INTERNAL_BOUNDS_CHECKS
        @assert axes(db) == (stage_axis,)
        @assert axes(b) == (stage_axis,)
    end

    # Construct numeric constants.
    _zero = zero(T)

    # Compute tolerance for comparison to zero.
    max_abs_residual = _zero
    for i in output_axis
        max_abs_residual = max(max_abs_residual,
            abs(compute_residual(adj.ev, b, i)))
    end
    pos_tolerance = eps(T) * max_abs_residual
    neg_tolerance = -pos_tolerance

    # Initialize db to zero.
    @simd ivdep for i in stage_axis
        @inbounds db[i] = _zero
    end

    # Compute db using derivative of weighted L1 norm.
    @inbounds for (i, k) in pairs(adj.ev.table.selected_indices)
        residual = compute_residual(adj.ev, b, i)
        if residual < neg_tolerance
            w = cost.weights[i]
            @simd ivdep for j in stage_axis
                db[j] -= w * adj.ev.Phi[j, k]
            end
        elseif residual > pos_tolerance
            w = cost.weights[i]
            @simd ivdep for j in stage_axis
                db[j] += w * adj.ev.Phi[j, k]
            end
        end
    end

    return db
end


############################################################################# L2


export RKCostL2


"""
```math
\\sum_{t \\in T} \\left(
\\mathbf{b} \\cdot \\Phi_t(A) - \\frac{1}{\\gamma(t)}
\\right)^2
```
"""
struct RKCostL2{T} <: AbstractRKCost{T} end


@inline RKCostL2() = RKCostL2{Float64}()


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


#################################################################### WEIGHTED L2


export RKCostWeightedL2


"""
```math
\\sum_{t \\in T} w_t \\left(
\\mathbf{b} \\cdot \\Phi_t(A) - \\frac{1}{\\gamma(t)}
\\right)^2
```
"""
struct RKCostWeightedL2{T} <: AbstractRKCost{T}
    weights::Vector{T}
end


function (cost::RKCostWeightedL2{T})(
    ev::AbstractRKOCEvaluator{T},
    b::AbstractVector{T},
) where {T}

    # Validate array dimensions.
    stage_axis, _, output_axis = get_axes(ev)
    @static if PERFORM_INTERNAL_BOUNDS_CHECKS
        @assert axes(cost.weights) == (output_axis,)
        @assert axes(b) == (stage_axis,)
    end

    # Compute weighted L2 norm of residuals.
    result = zero(T)
    @inbounds for i in output_axis
        result += cost.weights[i] * abs2(compute_residual(ev, b, i))
    end

    return result
end


function (cost::RKCostWeightedL2{T})(
    adj::AbstractRKOCAdjoint{T},
    b::AbstractVector{T},
) where {T}

    # Validate array dimensions.
    stage_axis, _, output_axis = get_axes(adj.ev)
    @static if PERFORM_INTERNAL_BOUNDS_CHECKS
        @assert axes(cost.weights) == (output_axis,)
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
            derivative = cost.weights[i] * (residual + residual)
            @simd ivdep for j in stage_axis
                adj.ev.dPhi[j, k] = derivative * b[j]
            end
        end
    end

    return adj
end


function (cost::RKCostWeightedL2{T})(
    db::AbstractVector{T},
    adj::AbstractRKOCAdjoint{T},
    b::AbstractVector{T},
) where {T}

    # Validate array dimensions.
    stage_axis, _, output_axis = get_axes(adj.ev)
    @static if PERFORM_INTERNAL_BOUNDS_CHECKS
        @assert axes(cost.weights) == (output_axis,)
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
        derivative = cost.weights[i] * (residual + residual)
        @simd ivdep for j in stage_axis
            db[j] += derivative * adj.ev.Phi[j, k]
        end
    end

    return db
end


##################################################################### L-INFINITY


export RKCostLInfinity


"""
```math
\\max_{t \\in T} \\left\\lvert
\\mathbf{b} \\cdot \\Phi_t(A) - \\frac{1}{\\gamma(t)}
\\right\\rvert
```
"""
struct RKCostLInfinity{T} <: AbstractRKCost{T} end


@inline RKCostLInfinity() = RKCostLInfinity{Float64}()


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


"""
```math
\\max_{t \\in T} w_t \\left\\lvert
\\mathbf{b} \\cdot \\Phi_t(A) - \\frac{1}{\\gamma(t)}
\\right\\rvert
```
"""
struct RKCostWeightedLInfinity{T} <: AbstractRKCost{T}
    weights::Vector{T}
end


function (cost::RKCostWeightedLInfinity{T})(
    ev::AbstractRKOCEvaluator{T},
    b::AbstractVector{T},
) where {T}

    # Validate array dimensions.
    stage_axis, _, output_axis = get_axes(ev)
    @static if PERFORM_INTERNAL_BOUNDS_CHECKS
        @assert axes(cost.weights) == (output_axis,)
        @assert axes(b) == (stage_axis,)
    end

    # Compute weighted LInfinity norm of residuals.
    result = zero(T)
    @inbounds for i in output_axis
        result = max(result, cost.weights[i] * abs(compute_residual(ev, b, i)))
    end

    return result
end


########################################################################## HUBER


export RKCostHuber


"""
```math
\\sum_{t \\in T} H_\\delta\\!\\left(
\\mathbf{b} \\cdot \\Phi_t(A) - \\frac{1}{\\gamma(t)}
\\right)
```
```math
H_\\delta(x) = \\begin{cases}
x^2 & \\text{if } \\lvert x \\rvert \\leq \\delta \\\\
2 \\delta \\lvert x \\rvert - \\delta^2 & \\text{otherwise}
\\end{cases}
```
"""
struct RKCostHuber{T} <: AbstractRKCost{T}
    delta::T
end


@inline function huber_loss(delta::T, x::T) where {T}
    abs_x = abs(x)
    return ((abs_x <= delta) ? (abs_x * abs_x) :
            (delta * ((abs_x + abs_x) - delta)))
end


@inline function huber_derivative(delta::T, x::T) where {T}
    if x > delta
        return delta + delta
    else
        neg_delta = -delta
        if x < neg_delta
            return neg_delta + neg_delta
        else
            return x + x
        end
    end
end


function (cost::RKCostHuber{T})(
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
        result += huber_loss(cost.delta, compute_residual(ev, b, i))
    end

    return result
end


function (cost::RKCostHuber{T})(
    adj::AbstractRKOCAdjoint{T},
    b::AbstractVector{T},
) where {T}

    # Validate array dimensions.
    stage_axis, _, output_axis = get_axes(adj.ev)
    @static if PERFORM_INTERNAL_BOUNDS_CHECKS
        @assert axes(b) == (stage_axis,)
    end

    # Construct numeric constants.
    _zero = zero(T)

    # Initialize dPhi using derivative of Huber loss.
    @inbounds for (k, i) in Iterators.reverse(pairs(adj.ev.table.source_indices))
        if i == NULL_INDEX
            @simd ivdep for j in stage_axis
                adj.ev.dPhi[j, k] = _zero
            end
        else
            derivative = huber_derivative(cost.delta,
                compute_residual(adj.ev, b, i))
            @simd ivdep for j in stage_axis
                adj.ev.dPhi[j, k] = derivative * b[j]
            end
        end
    end

    return adj
end


function (cost::RKCostHuber{T})(
    db::AbstractVector{T},
    adj::AbstractRKOCAdjoint{T},
    b::AbstractVector{T},
) where {T}

    # Validate array dimensions.
    stage_axis, _, output_axis = get_axes(adj.ev)
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

    # Compute db using derivative of weighted L2 norm.
    @inbounds for (i, k) in pairs(adj.ev.table.selected_indices)
        derivative = huber_derivative(cost.delta,
            compute_residual(adj.ev, b, i))
        @simd ivdep for j in stage_axis
            db[j] += derivative * adj.ev.Phi[j, k]
        end
    end

    return db
end


################################################################# WEIGHTED HUBER


export RKCostWeightedHuber


"""
```math
\\sum_{t \\in T} w_t H_\\delta\\!\\left(
\\mathbf{b} \\cdot \\Phi_t(A) - \\frac{1}{\\gamma(t)}
\\right)
```
```math
H_\\delta(x) = \\begin{cases}
x^2 & \\text{if } \\lvert x \\rvert \\leq \\delta \\\\
2 \\delta \\lvert x \\rvert - \\delta^2 & \\text{otherwise}
\\end{cases}
```
"""
struct RKCostWeightedHuber{T} <: AbstractRKCost{T}
    delta::T
    weights::Vector{T}
end


function (cost::RKCostWeightedHuber{T})(
    ev::AbstractRKOCEvaluator{T},
    b::AbstractVector{T}
) where {T}

    # Validate array dimensions.
    stage_axis, _, output_axis = get_axes(ev)
    @static if PERFORM_INTERNAL_BOUNDS_CHECKS
        @assert axes(cost.weights) == (output_axis,)
        @assert axes(b) == (stage_axis,)
    end

    # Compute weighted sum of Huber losses of residuals.
    result = zero(T)
    @inbounds for i in output_axis
        result += cost.weights[i] * huber_loss(cost.delta,
            compute_residual(ev, b, i))
    end

    return result
end


function (cost::RKCostWeightedHuber{T})(
    adj::AbstractRKOCAdjoint{T},
    b::AbstractVector{T},
) where {T}

    # Validate array dimensions.
    stage_axis, _, output_axis = get_axes(adj.ev)
    @static if PERFORM_INTERNAL_BOUNDS_CHECKS
        @assert axes(cost.weights) == (output_axis,)
        @assert axes(b) == (stage_axis,)
    end

    # Construct numeric constants.
    _zero = zero(T)

    # Initialize dPhi using derivative of Huber loss.
    @inbounds for (k, i) in Iterators.reverse(pairs(adj.ev.table.source_indices))
        if i == NULL_INDEX
            @simd ivdep for j in stage_axis
                adj.ev.dPhi[j, k] = _zero
            end
        else
            derivative = cost.weights[i] * huber_derivative(cost.delta,
                compute_residual(adj.ev, b, i))
            @simd ivdep for j in stage_axis
                adj.ev.dPhi[j, k] = derivative * b[j]
            end
        end
    end

    return adj
end


function (cost::RKCostWeightedHuber{T})(
    db::AbstractVector{T},
    adj::AbstractRKOCAdjoint{T},
    b::AbstractVector{T},
) where {T}

    # Validate array dimensions.
    stage_axis, _, output_axis = get_axes(adj.ev)
    @static if PERFORM_INTERNAL_BOUNDS_CHECKS
        @assert axes(cost.weights) == (output_axis,)
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
        derivative = cost.weights[i] * huber_derivative(cost.delta,
            compute_residual(adj.ev, b, i))
        @simd ivdep for j in stage_axis
            db[j] += derivative * adj.ev.Phi[j, k]
        end
    end

    return db
end


end # module RKCost
