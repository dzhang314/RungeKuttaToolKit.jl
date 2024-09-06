module RKParameterization


using ..RungeKuttaToolKit: PERFORM_INTERNAL_BOUNDS_CHECKS,
    AbstractRKParameterization, AbstractRKParameterizationQR


export AbstractRKParameterization, AbstractRKParameterizationQR


####################################################################### EXPLICIT


export RKParameterizationExplicit


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


#################################################################### EXPLICIT QR


export RKParameterizationExplicitQR


struct RKParameterizationExplicitQR{T} <: AbstractRKParameterizationQR{T}

    num_stages::Int
    num_variables::Int

    @inline RKParameterizationExplicitQR{T}(s::Integer) where {T} =
        new{T}(s, (s * (s - 1)) >> 1)
end


@inline RKParameterizationExplicitQR(s::Integer) =
    RKParameterizationExplicitQR{Float64}(s)


function (param::RKParameterizationExplicitQR{T})(
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


function (param::RKParameterizationExplicitQR{T})(
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


######################################################### DIAGONALLY IMPLICIT QR


export RKParameterizationDiagonallyImplicitQR


struct RKParameterizationDiagonallyImplicitQR{T} <:
       AbstractRKParameterizationQR{T}

    num_stages::Int
    num_variables::Int

    @inline RKParameterizationDiagonallyImplicitQR{T}(s::Integer) where {T} =
        new{T}(s, (s * (s + 1)) >> 1)
end


@inline RKParameterizationDiagonallyImplicitQR(s::Integer) =
    RKParameterizationDiagonallyImplicitQR{Float64}(s)


function (param::RKParameterizationDiagonallyImplicitQR{T})(
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


function (param::RKParameterizationDiagonallyImplicitQR{T})(
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


#################################################################### IMPLICIT QR


export RKParameterizationImplicitQR


struct RKParameterizationImplicitQR{T} <: AbstractRKParameterizationQR{T}

    num_stages::Int
    num_variables::Int

    @inline RKParameterizationImplicitQR{T}(s::Integer) where {T} =
        new{T}(s, s * s)
end


@inline RKParameterizationImplicitQR(s::Integer) =
    RKParameterizationImplicitQR{Float64}(s)


function (param::RKParameterizationImplicitQR{T})(
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


function (param::RKParameterizationImplicitQR{T})(
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

    # Construct numeric constants.
    _zero = zero(T)

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


end # module RKParameterization
