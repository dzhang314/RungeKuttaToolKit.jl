module L2Gradients


using ..RungeKuttaToolKit: NULL_INDEX, PERFORM_INTERNAL_BOUNDS_CHECKS,
    RKOCEvaluator, RKOCEvaluatorSIMD, RKOCEvaluatorMFV, RKOCAdjoint,
    get_axes, compute_residual,
    vload_vec, mfvload_vec, vstore_vec!, mfvstore_vec!
using ..RungeKuttaToolKit.RKCost: RKCostL2
using SIMD: Vec
using MultiFloats: MultiFloat, MultiFloatVec


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


function (::RKCostL2{MultiFloat{T,N}})(
    adj::RKOCAdjoint{MultiFloat{T,N},RKOCEvaluatorMFV{S,T,N}},
    b::AbstractVector{MultiFloat{T,N}},
) where {S,T,N}

    # Validate array dimensions.
    stage_axis, _, _ = get_axes(adj.ev)
    @static if PERFORM_INTERNAL_BOUNDS_CHECKS
        @assert axes(b) == (stage_axis,)
    end

    # Construct numeric constants.
    _zeros = zero(MultiFloatVec{S,T,N})

    # Load b into SIMD registers.
    vb = mfvload_vec(Val{S}(), b)

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


function (::RKCostL2{MultiFloat{T,N}})(
    db::AbstractVector{MultiFloat{T,N}},
    adj::RKOCAdjoint{MultiFloat{T,N},RKOCEvaluatorMFV{S,T,N}},
    b::AbstractVector{MultiFloat{T,N}},
) where {S,T,N}

    # Validate array dimensions.
    stage_axis, _, _ = get_axes(adj.ev)
    @static if PERFORM_INTERNAL_BOUNDS_CHECKS
        @assert axes(db) == (stage_axis,)
        @assert axes(b) == (stage_axis,)
    end

    # Initialize db to zero.
    vdb = zero(MultiFloatVec{S,T,N})

    # Compute db using derivative of L2 norm.
    @inbounds for (i, k) in pairs(adj.ev.table.selected_indices)
        residual = compute_residual(adj.ev, b, i)
        derivative = residual + residual
        vdb += derivative * adj.ev.Phi[k]
    end

    mfvstore_vec!(db, vdb)
    return db
end


end # module L2Gradients
