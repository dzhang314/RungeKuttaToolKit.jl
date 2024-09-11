module RKParameterization


using ..RungeKuttaToolKit: PERFORM_INTERNAL_BOUNDS_CHECKS,
    AbstractRKOCEvaluator, AbstractRKOCEvaluatorAO,
    AbstractRKParameterization, AbstractRKParameterizationAO,
    get_axes, compute_Phi!, pushforward_dPhi!, pushforward_dresiduals!


export AbstractRKParameterization, AbstractRKParameterizationAO


@inline (param::AbstractRKParameterization{T})(
    x::AbstractVector{T},
) where {T} = param(Matrix{T}(undef, param.num_stages, param.num_stages),
    Vector{T}(undef, param.num_stages), x)


@inline (param::AbstractRKParameterization{T})(
    A::AbstractMatrix{T},
    b::AbstractVector{T},
) where {T} = param(Vector{T}(undef, param.num_variables), A, b)


@inline (param::AbstractRKParameterizationAO{T})(
    x::AbstractVector{T},
) where {T} = param(Matrix{T}(undef, param.num_stages, param.num_stages), x)


@inline (param::AbstractRKParameterizationAO{T})(
    A::AbstractMatrix{T},
) where {T} = param(Vector{T}(undef, param.num_variables), A)


####################################################################### EXPLICIT


export RKParameterizationExplicit


"""
"""
struct RKParameterizationExplicit{T} <: AbstractRKParameterization{T}

    num_stages::Int
    num_variables::Int

    @inline RKParameterizationExplicit{T}(s::Integer) where {T} =
        new{T}(s, (s * (s + 1)) >> 1)
end


@inline RKParameterizationExplicit(s::Integer) =
    RKParameterizationExplicit{Float64}(s)


function (param::RKParameterizationExplicit{T})(
    A::AbstractMatrix{T},
    b::AbstractVector{T},
    x::AbstractVector{T},
) where {T}

    # Validate array dimensions.
    s = param.num_stages
    @static if PERFORM_INTERNAL_BOUNDS_CHECKS
        Base.require_one_based_indexing(A, b, x)
        @assert size(A) == (s, s)
        @assert size(b) == (s,)
        @assert size(x) == (param.num_variables,)
    end

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


function (param::RKParameterizationExplicit{T})(
    x::AbstractVector{T},
    A::AbstractMatrix{T},
    b::AbstractVector{T},
) where {T}

    # Validate array dimensions.
    s = param.num_stages
    @static if PERFORM_INTERNAL_BOUNDS_CHECKS
        Base.require_one_based_indexing(x, A, b)
        @assert size(x) == (param.num_variables,)
        @assert size(A) == (s, s)
        @assert size(b) == (s,)
    end

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


function (param::RKParameterizationExplicit{T})(
    jacobian::AbstractMatrix{T},
    A::AbstractMatrix{T},
    b::AbstractVector{T},
    ev::AbstractRKOCEvaluator{T},
    x::AbstractVector{T},
) where {T}

    # Validate array dimensions.
    s = param.num_stages
    stage_axis, _, output_axis = get_axes(ev)
    @static if PERFORM_INTERNAL_BOUNDS_CHECKS
        Base.require_one_based_indexing(jacobian, A, b, x)
        @assert size(jacobian) == (length(output_axis), param.num_variables)
        @assert size(A) == (s, s)
        @assert size(b) == (s,)
        @assert size(x) == (param.num_variables,)
        @assert size(stage_axis) == (s,)
    end

    # Compute Butcher weight vectors.
    param(A, b, x)
    compute_Phi!(ev, A)

    # Compute derivatives with respect to the strict lower-triangular part of A.
    k = 0
    for i = 1:s
        for j = 1:i-1
            pushforward_dPhi!(ev, A, i, j)
            pushforward_dresiduals!(view(jacobian, :, k += 1), ev, b)
        end
    end

    # Compute derivatives with respect to b.
    for i = 1:s
        k += 1
        @inbounds for (j, m) in pairs(ev.table.selected_indices)
            jacobian[j, k] = ev.Phi[i, m]
        end
    end

    return jacobian
end


#################################################################### EXPLICIT AO


export RKParameterizationExplicitAO


struct RKParameterizationExplicitAO{T} <: AbstractRKParameterizationAO{T}

    num_stages::Int
    num_variables::Int

    @inline RKParameterizationExplicitAO{T}(s::Integer) where {T} =
        new{T}(s, (s * (s - 1)) >> 1)
end


@inline RKParameterizationExplicitAO(s::Integer) =
    RKParameterizationExplicitAO{Float64}(s)


function (param::RKParameterizationExplicitAO{T})(
    A::AbstractMatrix{T},
    x::AbstractVector{T},
) where {T}

    # Validate array dimensions.
    s = param.num_stages
    @static if PERFORM_INTERNAL_BOUNDS_CHECKS
        Base.require_one_based_indexing(A, x)
        @assert size(A) == (s, s)
        @assert size(x) == (param.num_variables,)
    end

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


function (param::RKParameterizationExplicitAO{T})(
    x::AbstractVector{T},
    A::AbstractMatrix{T},
) where {T}

    # Validate array dimensions.
    s = param.num_stages
    @static if PERFORM_INTERNAL_BOUNDS_CHECKS
        Base.require_one_based_indexing(x, A)
        @assert size(x) == (param.num_variables,)
        @assert size(A) == (s, s)
    end

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


############################################################ DIAGONALLY IMPLICIT


export RKParameterizationDiagonallyImplicit


"""
"""
struct RKParameterizationDiagonallyImplicit{T} <: AbstractRKParameterization{T}

    num_stages::Int
    num_variables::Int

    @inline RKParameterizationDiagonallyImplicit{T}(s::Integer) where {T} =
        new{T}(s, (s * (s + 3)) >> 1)
end


@inline RKParameterizationDiagonallyImplicit(s::Integer) =
    RKParameterizationDiagonallyImplicit{Float64}(s)


function (param::RKParameterizationDiagonallyImplicit{T})(
    A::AbstractMatrix{T},
    b::AbstractVector{T},
    x::AbstractVector{T},
) where {T}

    # Validate array dimensions.
    s = param.num_stages
    @static if PERFORM_INTERNAL_BOUNDS_CHECKS
        Base.require_one_based_indexing(A, b, x)
        @assert size(A) == (s, s)
        @assert size(b) == (s,)
        @assert size(x) == (param.num_variables,)
    end

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


function (param::RKParameterizationDiagonallyImplicit{T})(
    x::AbstractVector{T},
    A::AbstractMatrix{T},
    b::AbstractVector{T},
) where {T}

    # Validate array dimensions.
    s = param.num_stages
    @static if PERFORM_INTERNAL_BOUNDS_CHECKS
        Base.require_one_based_indexing(x, A, b)
        @assert size(x) == (param.num_variables,)
        @assert size(A) == (s, s)
        @assert size(b) == (s,)
    end

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


function (param::RKParameterizationDiagonallyImplicit{T})(
    jacobian::AbstractMatrix{T},
    A::AbstractMatrix{T},
    b::AbstractVector{T},
    ev::AbstractRKOCEvaluator{T},
    x::AbstractVector{T},
) where {T}

    # Validate array dimensions.
    s = param.num_stages
    stage_axis, _, output_axis = get_axes(ev)
    @static if PERFORM_INTERNAL_BOUNDS_CHECKS
        Base.require_one_based_indexing(jacobian, A, b, x)
        @assert size(jacobian) == (length(output_axis), param.num_variables)
        @assert size(A) == (s, s)
        @assert size(b) == (s,)
        @assert size(x) == (param.num_variables,)
        @assert size(stage_axis) == (s,)
    end

    # Compute Butcher weight vectors.
    param(A, b, x)
    compute_Phi!(ev, A)

    # Compute derivatives with respect to the lower-triangular part of A.
    k = 0
    for i = 1:s
        for j = 1:i
            pushforward_dPhi!(ev, A, i, j)
            pushforward_dresiduals!(view(jacobian, :, k += 1), ev, b)
        end
    end

    # Compute derivatives with respect to b.
    for i = 1:s
        k += 1
        @inbounds for (j, m) in pairs(ev.table.selected_indices)
            jacobian[j, k] = ev.Phi[i, m]
        end
    end

    return jacobian
end


######################################################### DIAGONALLY IMPLICIT AO


export RKParameterizationDiagonallyImplicitAO


struct RKParameterizationDiagonallyImplicitAO{T} <:
       AbstractRKParameterizationAO{T}

    num_stages::Int
    num_variables::Int

    @inline RKParameterizationDiagonallyImplicitAO{T}(s::Integer) where {T} =
        new{T}(s, (s * (s + 1)) >> 1)
end


@inline RKParameterizationDiagonallyImplicitAO(s::Integer) =
    RKParameterizationDiagonallyImplicitAO{Float64}(s)


function (param::RKParameterizationDiagonallyImplicitAO{T})(
    A::AbstractMatrix{T},
    x::AbstractVector{T},
) where {T}

    # Validate array dimensions.
    s = param.num_stages
    @static if PERFORM_INTERNAL_BOUNDS_CHECKS
        Base.require_one_based_indexing(A, x)
        @assert size(A) == (s, s)
        @assert size(x) == (param.num_variables,)
    end

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


function (param::RKParameterizationDiagonallyImplicitAO{T})(
    x::AbstractVector{T},
    A::AbstractMatrix{T},
) where {T}

    # Validate array dimensions.
    s = param.num_stages
    @static if PERFORM_INTERNAL_BOUNDS_CHECKS
        Base.require_one_based_indexing(x, A)
        @assert size(x) == (param.num_variables,)
        @assert size(A) == (s, s)
    end

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


####################################################################### IMPLICIT


export RKParameterizationImplicit


"""
"""
struct RKParameterizationImplicit{T} <: AbstractRKParameterization{T}

    num_stages::Int
    num_variables::Int

    @inline RKParameterizationImplicit{T}(s::Integer) where {T} =
        new{T}(s, s * (s + 1))
end


@inline RKParameterizationImplicit(s::Integer) =
    RKParameterizationImplicit{Float64}(s)


function (param::RKParameterizationImplicit{T})(
    A::AbstractMatrix{T},
    b::AbstractVector{T},
    x::AbstractVector{T},
) where {T}

    # Validate array dimensions.
    s = param.num_stages
    @static if PERFORM_INTERNAL_BOUNDS_CHECKS
        Base.require_one_based_indexing(A, b, x)
        @assert size(A) == (s, s)
        @assert size(b) == (s,)
        @assert size(x) == (param.num_variables,)
    end

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


function (param::RKParameterizationImplicit{T})(
    x::AbstractVector{T},
    A::AbstractMatrix{T},
    b::AbstractVector{T},
) where {T}

    # Validate array dimensions.
    s = param.num_stages
    @static if PERFORM_INTERNAL_BOUNDS_CHECKS
        Base.require_one_based_indexing(x, A, b)
        @assert size(x) == (param.num_variables,)
        @assert size(A) == (s, s)
        @assert size(b) == (s,)
    end

    # Iterate over A.
    offset = 0
    for i = 1:s
        @simd ivdep for j = 1:s
            @inbounds x[offset+j] = A[i, j]
        end
        offset += s
    end

    # Iterate over b.
    @simd ivdep for i = 1:s
        @inbounds x[offset+i] = b[i]
    end

    return x
end


function (param::RKParameterizationImplicit{T})(
    jacobian::AbstractMatrix{T},
    A::AbstractMatrix{T},
    b::AbstractVector{T},
    ev::AbstractRKOCEvaluator{T},
    x::AbstractVector{T},
) where {T}

    # Validate array dimensions.
    s = param.num_stages
    stage_axis, _, output_axis = get_axes(ev)
    @static if PERFORM_INTERNAL_BOUNDS_CHECKS
        Base.require_one_based_indexing(jacobian, A, b, x)
        @assert size(jacobian) == (length(output_axis), param.num_variables)
        @assert size(A) == (s, s)
        @assert size(b) == (s,)
        @assert size(x) == (param.num_variables,)
        @assert size(stage_axis) == (s,)
    end

    # Compute Butcher weight vectors.
    param(A, b, x)
    compute_Phi!(ev, A)

    # Compute derivatives with respect to the lower-triangular part of A.
    k = 0
    for i = 1:s
        for j = 1:s
            pushforward_dPhi!(ev, A, i, j)
            pushforward_dresiduals!(view(jacobian, :, k += 1), ev, b)
        end
    end

    # Compute derivatives with respect to b.
    for i = 1:s
        k += 1
        @inbounds for (j, m) in pairs(ev.table.selected_indices)
            jacobian[j, k] = ev.Phi[i, m]
        end
    end

    return jacobian
end


#################################################################### IMPLICIT AO


export RKParameterizationImplicitAO


struct RKParameterizationImplicitAO{T} <: AbstractRKParameterizationAO{T}

    num_stages::Int
    num_variables::Int

    @inline RKParameterizationImplicitAO{T}(s::Integer) where {T} =
        new{T}(s, s * s)
end


@inline RKParameterizationImplicitAO(s::Integer) =
    RKParameterizationImplicitAO{Float64}(s)


function (param::RKParameterizationImplicitAO{T})(
    A::AbstractMatrix{T},
    x::AbstractVector{T},
) where {T}

    # Validate array dimensions.
    s = param.num_stages
    @static if PERFORM_INTERNAL_BOUNDS_CHECKS
        Base.require_one_based_indexing(A, x)
        @assert size(A) == (s, s)
        @assert size(x) == (param.num_variables,)
    end

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


function (param::RKParameterizationImplicitAO{T})(
    x::AbstractVector{T},
    A::AbstractMatrix{T},
) where {T}

    # Validate array dimensions.
    s = param.num_stages
    @static if PERFORM_INTERNAL_BOUNDS_CHECKS
        Base.require_one_based_indexing(x, A)
        @assert size(x) == (param.num_variables,)
        @assert size(A) == (s, s)
    end

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


############################################################## PARALLEL EXPLICIT


export RKParameterizationParallelExplicit


"""
"""
struct RKParameterizationParallelExplicit{T} <: AbstractRKParameterization{T}
    num_stages::Int
    num_variables::Int
    num_parallel_stages::Int
    parallel_width::Int

    @inline RKParameterizationParallelExplicit{T}(
        num_parallel_stages::Integer,
        parallel_width::Integer,
    ) where {T} = new{T}(
        1 + num_parallel_stages * parallel_width,
        1 + ((num_parallel_stages * parallel_width *
              (4 + (num_parallel_stages - 1) * parallel_width)) >> 1),
        num_parallel_stages, parallel_width)
end


@inline RKParameterizationParallelExplicit(
    num_parallel_stages::Integer,
    parallel_width::Integer,
) = RKParameterizationParallelExplicit{Float64}(
    num_parallel_stages, parallel_width)


function (param::RKParameterizationParallelExplicit{T})(
    A::AbstractMatrix{T},
    b::AbstractVector{T},
    x::AbstractVector{T},
) where {T}

    # Validate array dimensions.
    s = param.num_stages
    @static if PERFORM_INTERNAL_BOUNDS_CHECKS
        Base.require_one_based_indexing(A, b, x)
        @assert size(A) == (s, s)
        @assert size(b) == (s,)
        @assert size(x) == (param.num_variables,)
    end

    # Construct numeric constants.
    _zero = zero(T)

    # Initialize first row of A.
    i = 1
    @simd ivdep for j = 1:s
        @inbounds A[i, j] = _zero
    end

    # Iterate over lower-triangular blocks of A.
    offset = 0
    for parallel_stage = 1:param.num_parallel_stages
        row_length = 1 + (parallel_stage - 1) * param.parallel_width
        for _ = 1:param.parallel_width
            i += 1
            @simd ivdep for j = 1:row_length
                @inbounds A[i, j] = x[offset+j]
            end
            @simd ivdep for j = row_length+1:s
                @inbounds A[i, j] = _zero
            end
            offset += row_length
        end
    end

    # Iterate over b.
    @simd ivdep for i = 1:s
        @inbounds b[i] = x[offset+i]
    end

    return (A, b)
end


function (param::RKParameterizationParallelExplicit{T})(
    x::AbstractVector{T},
    A::AbstractMatrix{T},
    b::AbstractVector{T},
) where {T}

    # Validate array dimensions.
    s = param.num_stages
    @static if PERFORM_INTERNAL_BOUNDS_CHECKS
        Base.require_one_based_indexing(x, A, b)
        @assert size(x) == (param.num_variables,)
        @assert size(A) == (s, s)
        @assert size(b) == (s,)
    end

    # Iterate over lower-triangular blocks of A.
    i = 1
    offset = 0
    for parallel_stage = 1:param.num_parallel_stages
        row_length = 1 + (parallel_stage - 1) * param.parallel_width
        for _ = 1:param.parallel_width
            i += 1
            @simd ivdep for j = 1:row_length
                @inbounds x[offset+j] = A[i, j]
            end
            offset += row_length
        end
    end

    # Iterate over b.
    @simd ivdep for i = 1:s
        @inbounds x[offset+i] = b[i]
    end

    return x
end


function (param::RKParameterizationParallelExplicit{T})(
    jacobian::AbstractMatrix{T},
    A::AbstractMatrix{T},
    b::AbstractVector{T},
    ev::AbstractRKOCEvaluator{T},
    x::AbstractVector{T},
) where {T}

    # Validate array dimensions.
    s = param.num_stages
    stage_axis, _, output_axis = get_axes(ev)
    @static if PERFORM_INTERNAL_BOUNDS_CHECKS
        Base.require_one_based_indexing(jacobian, A, b, x)
        @assert size(jacobian) == (length(output_axis), param.num_variables)
        @assert size(A) == (s, s)
        @assert size(b) == (s,)
        @assert size(x) == (param.num_variables,)
        @assert size(stage_axis) == (s,)
    end

    # Compute Butcher weight vectors.
    param(A, b, x)
    compute_Phi!(ev, A)

    # Compute derivatives with respect to lower-triangular blocks of A.
    i = 1
    k = 0
    for parallel_stage = 1:param.num_parallel_stages
        row_length = 1 + (parallel_stage - 1) * param.parallel_width
        for _ = 1:param.parallel_width
            i += 1
            for j = 1:row_length
                pushforward_dPhi!(ev, A, i, j)
                pushforward_dresiduals!(view(jacobian, :, k += 1), ev, b)
            end
        end
    end

    # Compute derivatives with respect to b.
    for i = 1:s
        k += 1
        @inbounds for (j, m) in pairs(ev.table.selected_indices)
            jacobian[j, k] = ev.Phi[i, m]
        end
    end

    return jacobian
end


end # module RKParameterization
