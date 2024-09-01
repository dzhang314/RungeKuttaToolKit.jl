module RKParameterization


using ..RungeKuttaToolKit: PERFORM_INTERNAL_BOUNDS_CHECKS,
    AbstractRKParameterization


####################################################################### EXPLICIT


export RKParameterizationExplicit


struct RKParameterizationExplicit{T} <: AbstractRKParameterization{T}

    num_stages::Int
    num_variables::Int

    @inline RKParameterizationExplicit{T}(s::Int) where {T} =
        new{T}(s, (s * (s + 1)) >> 1)
end


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


struct RKParameterizationExplicitQR{T} <: AbstractRKParameterization{T}

    num_stages::Int
    num_variables::Int

    @inline RKParameterizationExplicitQR{T}(s::Int) where {T} =
        new{T}(s, (s * (s - 1)) >> 1)
end


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

    @inline RKParameterizationDiagonallyImplicit{T}(s::Int) where {T} =
        new{T}(s, (s * (s + 3)) >> 1)
end


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
       AbstractRKParameterization{T}

    num_stages::Int
    num_variables::Int

    @inline RKParameterizationDiagonallyImplicitQR{T}(s::Int) where {T} =
        new{T}(s, (s * (s + 1)) >> 1)
end


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

    @inline RKParameterizationImplicit{T}(s::Int) where {T} =
        new{T}(s, s * (s + 1))
end


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
    copy!
    @simd ivdep for i = 1:s
        @inbounds x[offset+i] = b[i]
    end

    return x
end


#################################################################### IMPLICIT QR


export RKParameterizationImplicitQR


struct RKParameterizationImplicitQR{T} <: AbstractRKParameterization{T}

    num_stages::Int
    num_variables::Int

    @inline RKParameterizationImplicitQR{T}(s::Int) where {T} =
        new{T}(s, s * s)
end


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


struct RKParameterizationParallelExplicit{T} <: AbstractRKParameterization{T}
    num_stages::Int
    num_variables::Int
    num_parallel_stages::Int
    parallel_width::Int
end


end # module RKParameterization
