module RungeKuttaToolKit


using MultiFloats: MultiFloat, rsqrt


include("ButcherInstructions.jl")
using .ButcherInstructions: LevelSequence, ButcherInstruction,
    all_rooted_trees, butcher_density, butcher_symmetry,
    build_instruction_table, compute_children_siblings


####################################################### EVALUATOR DATA STRUCTURE


export RKOCEvaluator


struct RKOCEvaluator{T}

    instructions::Vector{ButcherInstruction}
    output_indices::Vector{Int}
    source_indices::Vector{Int}

    child_indices::Vector{Int}
    sibling_indices::Vector{Pair{Int,Int}}
    sibling_ranges::Vector{UnitRange{Int}}

    A::Matrix{T}
    dA::Matrix{T}
    b::Vector{T}
    db::Vector{T}
    phi::Matrix{T}
    dphi::Matrix{T}
    inv_gamma::Vector{T}
    residuals::Vector{T}

end


function RKOCEvaluator{T}(trees::Vector{LevelSequence}, s::Int) where {T}

    instructions, output_indices = build_instruction_table(trees)
    source_indices = [-1 for _ in instructions]
    for (i, j) in enumerate(output_indices)
        source_indices[j] = i
    end

    child_indices, sibling_lists = compute_children_siblings(instructions)
    sibling_indices = reduce(vcat, sibling_lists)
    end_indices = cumsum(length.(sibling_lists))
    start_indices = vcat([1], end_indices[1:end-1] .+ 1)
    sibling_ranges = UnitRange{Int}.(start_indices, end_indices)

    return RKOCEvaluator{T}(
        instructions, output_indices, source_indices,
        child_indices, sibling_indices, sibling_ranges,
        Matrix{T}(undef, s, s), Matrix{T}(undef, s, s),
        Vector{T}(undef, s), Vector{T}(undef, s),
        Matrix{T}(undef, s, length(instructions)),
        Matrix{T}(undef, s, length(instructions)),
        [inv(T(butcher_density(tree))) for tree in trees],
        Vector{T}(undef, length(trees)))
end


@inline RKOCEvaluator{T}(p::Int, s::Int) where {T} =
    RKOCEvaluator{T}(all_rooted_trees(p), s)


############################################ PROJECTION EVALUATOR DATA STRUCTURE


export RKOCProjectionEvaluator


struct RKOCProjectionEvaluator{T}

    instructions::Vector{ButcherInstruction}
    output_indices::Vector{Int}
    source_indices::Vector{Int}

    child_indices::Vector{Int}
    sibling_indices::Vector{Pair{Int,Int}}
    sibling_ranges::Vector{UnitRange{Int}}

    A::Matrix{T}
    dA::Matrix{T}
    b::Vector{T}
    phi::Matrix{T}
    dphi::Matrix{T}
    inv_gamma::Vector{T}
    residuals::Vector{T}

    Q::Matrix{T}
    R::Matrix{T}

end


function RKOCProjectionEvaluator{T}(
    trees::Vector{LevelSequence}, s::Int
) where {T}

    instructions, output_indices = build_instruction_table(trees)
    source_indices = [-1 for _ in instructions]
    for (i, j) in enumerate(output_indices)
        source_indices[j] = i
    end

    child_indices, sibling_lists = compute_children_siblings(instructions)
    sibling_indices = reduce(vcat, sibling_lists)
    end_indices = cumsum(length.(sibling_lists))
    start_indices = vcat([1], end_indices[1:end-1] .+ 1)
    sibling_ranges = UnitRange{Int}.(start_indices, end_indices)

    return RKOCProjectionEvaluator{T}(
        instructions, output_indices, source_indices,
        child_indices, sibling_indices, sibling_ranges,
        Matrix{T}(undef, s, s), Matrix{T}(undef, s, s),
        Vector{T}(undef, s),
        Matrix{T}(undef, s, length(instructions)),
        Matrix{T}(undef, s, length(instructions)),
        [inv(T(butcher_density(tree))) for tree in trees],
        Vector{T}(undef, length(trees)),
        Matrix{T}(undef, length(trees), s),
        Matrix{T}(undef, s, s))
end


@inline RKOCProjectionEvaluator{T}(p::Int, s::Int) where {T} =
    RKOCProjectionEvaluator{T}(all_rooted_trees(p), s)


############################################ RESIDUAL COMPUTATION (FORWARD PASS)


function compute_phi!(
    phi::AbstractMatrix{T}, A::AbstractMatrix{T},
    instructions::Vector{ButcherInstruction}
) where {T}
    s = size(phi, 1)
    @assert (s, s) == size(A)
    @assert size(phi, 2) == length(instructions)
    _zero = zero(T)
    _one = one(T)
    @inbounds for (k, instruction) in enumerate(instructions)
        p, q = instruction.left, instruction.right
        if p == -1
            @assert q == -1
            @simd ivdep for j = 1:s
                phi[j, k] = _one
            end
        elseif q == -1
            @simd ivdep for j = 1:s
                phi[j, k] = _zero
            end
            for i = 1:s
                temp = phi[i, p]
                @simd ivdep for j = 1:s
                    phi[j, k] += temp * A[j, i]
                end
            end
        else
            @simd ivdep for j = 1:s
                phi[j, k] = phi[j, p] * phi[j, q]
            end
        end
    end
    return phi
end


function compute_residuals!(
    residuals::AbstractVector{T}, b::AbstractVector{T},
    phi::AbstractMatrix{T}, inv_gamma::AbstractVector{T},
    output_indices::AbstractVector{Int}
) where {T}
    t = length(residuals)
    s = length(b)
    @assert s == size(phi, 1)
    @assert (t,) == size(inv_gamma)
    @assert (t,) == size(output_indices)
    _zero = zero(T)
    @inbounds for (i, k) in enumerate(output_indices)
        result = _zero
        for j = 1:s
            result += b[j] * phi[j, k]
        end
        residuals[i] = result - inv_gamma[i]
    end
    return residuals
end


function compute_residuals!(
    residuals::AbstractVector{T}, Q::AbstractMatrix{T},
    inv_gamma::AbstractVector{T}
) where {T}
    t = length(residuals)
    s = size(Q, 2)
    @assert t == size(Q, 1)
    @assert (t,) == size(inv_gamma)
    _zero = zero(T)
    @simd ivdep for i = 1:t
        @inbounds residuals[i] = -inv_gamma[i]
    end
    @inbounds for j = 1:s
        overlap = _zero
        for i = 1:t
            overlap += Q[i, j] * residuals[i]
        end
        @simd ivdep for i = 1:t
            residuals[i] -= overlap * Q[i, j]
        end
    end
    return residuals
end


function compute_residuals!(
    residuals::AbstractVector{T}, b::AbstractVector{T},
    Q::AbstractMatrix{T}, inv_gamma::AbstractVector{T}
) where {T}
    t = length(residuals)
    s = length(b)
    @assert (t, s) == size(Q)
    @assert (t,) == size(inv_gamma)
    _zero = zero(T)
    @simd ivdep for i = 1:t
        @inbounds residuals[i] = -inv_gamma[i]
    end
    @inbounds for j = 1:s
        overlap = _zero
        for i = 1:t
            overlap += Q[i, j] * residuals[i]
        end
        @simd ivdep for i = 1:t
            residuals[i] -= overlap * Q[i, j]
        end
        b[j] = -overlap
    end
    return (residuals, b)
end


########################################################### LEAST-SQUARES SOLVER


@inline inv_sqrt(x::Float16) = rsqrt(x)
@inline inv_sqrt(x::Float32) = rsqrt(x)
@inline inv_sqrt(x::Float64) = rsqrt(x)
@inline inv_sqrt(x::MultiFloat{T,N}) where {T,N} = rsqrt(x)
@inline inv_sqrt(x::T) where {T} = inv(sqrt(x))


function gram_schmidt_qr!(Q::AbstractMatrix{T}) where {T}
    t, s = size(Q)
    _zero = zero(T)
    @inbounds for i = 1:s
        squared_norm = _zero
        for k = 1:t
            squared_norm += abs2(Q[k, i])
        end
        if !iszero(squared_norm)
            inv_norm = inv_sqrt(squared_norm)
            @simd ivdep for k = 1:t
                Q[k, i] *= inv_norm
            end
            for j = i+1:s
                overlap = _zero
                for k = 1:t
                    overlap += Q[k, i] * Q[k, j]
                end
                @simd ivdep for k = 1:t
                    Q[k, j] -= overlap * Q[k, i]
                end
            end
        end
    end
    return Q
end


function gram_schmidt_qr!(Q::AbstractMatrix{T}, R::AbstractMatrix{T}) where {T}
    t, s = size(Q)
    @assert (s, s) == size(R)
    # NOTE: R is stored transposed, and its diagonal is stored inverted.
    _zero = zero(T)
    @inbounds for i = 1:s
        squared_norm = _zero
        for k = 1:t
            squared_norm += abs2(Q[k, i])
        end
        if !iszero(squared_norm)
            inv_norm = inv_sqrt(squared_norm)
            R[i, i] = inv_norm
            @simd ivdep for k = 1:t
                Q[k, i] *= inv_norm
            end
            for j = i+1:s
                overlap = _zero
                for k = 1:t
                    overlap += Q[k, i] * Q[k, j]
                end
                R[j, i] = overlap
                @simd ivdep for k = 1:t
                    Q[k, j] -= overlap * Q[k, i]
                end
            end
        else
            R[i, i] = _zero
        end
    end
    return (Q, R)
end


function solve_upper_triangular!(
    b::AbstractVector{T}, R::AbstractMatrix{T}
) where {T}
    s = length(b)
    @assert (s, s) == size(R)
    _zero = zero(T)
    @inbounds for i = s:-1:1
        if iszero(R[i, i])
            b[i] = _zero
        else
            overlap = _zero
            for j = i+1:s
                overlap += R[j, i] * b[j]
            end
            b[i] = R[i, i] * (b[i] - overlap)
        end
    end
    return b
end



########################################### GRADIENT COMPUTATION (BACKWARD PASS)


function initialize_dphi_from_residual!(
    dphi::AbstractMatrix{T}, b::AbstractVector{T},
    residuals::AbstractVector{T}, source_indices::AbstractVector{Int}
) where {T}
    n, m = size(dphi)
    @assert (n,) == size(b)
    @assert (m,) == size(source_indices)
    _zero = zero(T)
    @inbounds for (k, s) in Iterators.reverse(enumerate(source_indices))
        if s == -1
            @simd ivdep for j = 1:n
                dphi[j, k] = _zero
            end
        else
            residual = residuals[s]
            twice_residual = residual + residual
            @simd ivdep for j = 1:n
                dphi[j, k] = twice_residual * b[j]
            end
        end
    end
    return dphi
end


function compute_dphi_backward!(
    dphi::AbstractMatrix{T}, A::AbstractMatrix{T}, phi::AbstractMatrix{T},
    child_indices::AbstractVector{Int},
    sibling_ranges::AbstractVector{UnitRange{Int}},
    sibling_indices::AbstractVector{Pair{Int,Int}}
) where {T}
    n, m = size(dphi)
    @assert (n, m) == size(phi)
    @assert (n, n) == size(A)
    @assert (m,) == size(child_indices)
    @assert (m,) == size(sibling_ranges)
    @inbounds for k = m:-1:1
        c = child_indices[k]
        if c != -1
            for j = 1:n
                temp = dphi[j, k]
                for i = 1:n
                    temp += A[i, j] * dphi[i, c]
                end
                dphi[j, k] = temp
            end
        end
        for i in sibling_ranges[k]
            (p, q) = sibling_indices[i]
            @simd ivdep for j = 1:n
                dphi[j, k] += phi[j, p] * dphi[j, q]
            end
        end
    end
    return dphi
end


function compute_dA_backward!(
    dA::AbstractMatrix{T}, phi::AbstractMatrix{T}, dphi::AbstractMatrix{T},
    child_indices::AbstractVector{Int}
) where {T}
    n, m = size(phi)
    @assert (n, m) == size(dphi)
    @assert (n, n) == size(dA)
    @assert (m,) == size(child_indices)
    _zero = zero(T)
    @inbounds for j = 1:n
        @simd ivdep for i = 1:n
            dA[i, j] = _zero
        end
    end
    @inbounds for (k, c) in enumerate(child_indices)
        if c != -1
            for t = 1:n
                f = phi[t, k]
                @simd ivdep for s = 1:n
                    dA[s, t] += f * dphi[s, c]
                end
            end
        end
    end
    return dA
end


function compute_db_backward!(
    db::AbstractVector{T}, phi::AbstractMatrix{T},
    residuals::AbstractVector{T}, output_indices::AbstractVector{Int}
) where {T}
    n = length(db)
    m = length(residuals)
    @assert n == size(phi, 1)
    @assert m == length(output_indices)
    _zero = zero(T)
    @inbounds begin
        @simd ivdep for i = 1:n
            db[i] = _zero
        end
        for (i, k) in enumerate(output_indices)
            r = residuals[i]
            @simd ivdep for j = 1:n
                db[j] += r * phi[j, k]
            end
        end
        @simd ivdep for i = 1:n
            db[i] += db[i]
        end
    end
    return db
end


################################################### RESHAPING COEFFICIENT ARRAYS


function reshape_explicit!(A::Matrix{T}, x::Vector{T}) where {T}
    s = size(A, 1)
    @assert s == size(A, 2)
    @assert ((s * (s - 1)) >> 1,) == size(x)
    _zero = zero(T)
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


function reshape_explicit!(x::Vector{T}, A::Matrix{T}) where {T}
    s = size(A, 1)
    @assert s == size(A, 2)
    @assert ((s * (s - 1)) >> 1,) == size(x)
    offset = 0
    for i = 2:s
        @simd ivdep for j = 1:i-1
            @inbounds x[offset+j] = A[i, j]
        end
        offset += i - 1
    end
    return x
end


function reshape_explicit!(A::Matrix{T}, b::Vector{T}, x::Vector{T}) where {T}
    s = length(b)
    @assert (s, s) == size(A)
    @assert ((s * (s + 1)) >> 1,) == size(x)
    _zero = zero(T)
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
    @simd ivdep for i = 1:s
        @inbounds b[i] = x[offset+i]
    end
    return (A, b)
end


function reshape_explicit!(x::Vector{T}, A::Matrix{T}, b::Vector{T}) where {T}
    s = length(b)
    @assert (s, s) == size(A)
    @assert ((s * (s + 1)) >> 1,) == size(x)
    offset = 0
    for i = 2:s
        @simd ivdep for j = 1:i-1
            @inbounds x[offset+j] = A[i, j]
        end
        offset += i - 1
    end
    @simd ivdep for i = 1:s
        @inbounds x[offset+i] = b[i]
    end
    return x
end


function reshape_implicit!(A::Matrix{T}, b::Vector{T}, x::Vector{T}) where {T}
    s = length(b)
    @assert (s, s) == size(A)
    @assert (s * (s + 1),) == size(x)
    n_squared = s * s
    @simd ivdep for i = 1:n_squared
        @inbounds A[i] = x[i]
    end
    @simd ivdep for i = 1:s
        @inbounds b[i] = x[n_squared+i]
    end
    return (A, b)
end


function reshape_implicit!(x::Vector{T}, A::Matrix{T}, b::Vector{T}) where {T}
    s = length(b)
    @assert (s, s) == size(A)
    @assert (s * (s + 1),) == size(x)
    n_squared = s * s
    @simd ivdep for i = 1:n_squared
        @inbounds x[i] = A[i]
    end
    @simd ivdep for i = 1:s
        @inbounds x[n_squared+i] = b[i]
    end
    return x
end


function populate_Q!(
    Q::AbstractMatrix{T}, phi::AbstractMatrix{T},
    output_indices::AbstractVector{Int}
) where {T}
    t, s = size(Q)
    @assert s == size(phi, 1)
    @assert (t,) == size(output_indices)
    for (i, k) in enumerate(output_indices)
        @simd ivdep for j = 1:s
            @inbounds Q[i, j] = phi[j, k]
        end
    end
end


############################################################## FUNCTOR INTERFACE


function (ev::RKOCEvaluator{T})(x::Vector{T}) where {T}
    reshape_explicit!(ev.A, ev.b, x)
    compute_phi!(ev.phi, ev.A, ev.instructions)
    compute_residuals!(ev.residuals,
        ev.b, ev.phi, ev.inv_gamma, ev.output_indices)
    result = zero(T)
    for i = 1:length(ev.residuals)
        residual = @inbounds ev.residuals[i]
        result += residual * residual
    end
    return result
end


function (proj::RKOCProjectionEvaluator{T})(x::Vector{T}) where {T}
    reshape_explicit!(proj.A, x)
    compute_phi!(proj.phi, proj.A, proj.instructions)
    populate_Q!(proj.Q, proj.phi, proj.output_indices)
    gram_schmidt_qr!(proj.Q)
    compute_residuals!(proj.residuals, proj.Q, proj.inv_gamma)
    result = zero(T)
    for i = 1:length(proj.residuals)
        residual = @inbounds proj.residuals[i]
        result += residual * residual
    end
    return result
end


struct RKOCEvaluatorAdjoint{T}
    ev::RKOCEvaluator{T}
end


@inline Base.adjoint(ev::RKOCEvaluator{T}) where {T} =
    RKOCEvaluatorAdjoint{T}(ev)


function (adj::RKOCEvaluatorAdjoint{T})(g::Vector{T}, x::Vector{T}) where {T}
    reshape_explicit!(adj.ev.A, adj.ev.b, x)
    compute_phi!(adj.ev.phi, adj.ev.A, adj.ev.instructions)
    compute_residuals!(adj.ev.residuals,
        adj.ev.b, adj.ev.phi, adj.ev.inv_gamma, adj.ev.output_indices)
    initialize_dphi_from_residual!(adj.ev.dphi,
        adj.ev.b, adj.ev.residuals, adj.ev.source_indices)
    compute_dphi_backward!(adj.ev.dphi, adj.ev.A, adj.ev.phi,
        adj.ev.child_indices, adj.ev.sibling_ranges, adj.ev.sibling_indices)
    compute_dA_backward!(adj.ev.dA,
        adj.ev.phi, adj.ev.dphi, adj.ev.child_indices)
    compute_db_backward!(adj.ev.db,
        adj.ev.phi, adj.ev.residuals, adj.ev.output_indices)
    reshape_explicit!(g, adj.ev.dA, adj.ev.db)
    return g
end


@inline (adj::RKOCEvaluatorAdjoint{T})(x::Vector{T}) where {T} =
    adj(similar(x), x)


struct RKOCProjectionAdjoint{T}
    proj::RKOCProjectionEvaluator{T}
end


@inline Base.adjoint(proj::RKOCProjectionEvaluator{T}) where {T} =
    RKOCProjectionAdjoint{T}(proj)


function (adj::RKOCProjectionAdjoint{T})(g::Vector{T}, x::Vector{T}) where {T}
    reshape_explicit!(adj.proj.A, x)
    compute_phi!(adj.proj.phi, adj.proj.A, adj.proj.instructions)
    populate_Q!(adj.proj.Q, adj.proj.phi, adj.proj.output_indices)
    gram_schmidt_qr!(adj.proj.Q, adj.proj.R)
    compute_residuals!(adj.proj.residuals, adj.proj.b,
        adj.proj.Q, adj.proj.inv_gamma)
    solve_upper_triangular!(adj.proj.b, adj.proj.R)
    initialize_dphi_from_residual!(adj.proj.dphi,
        adj.proj.b, adj.proj.residuals, adj.proj.source_indices)
    compute_dphi_backward!(adj.proj.dphi,
        adj.proj.A, adj.proj.phi, adj.proj.child_indices,
        adj.proj.sibling_ranges, adj.proj.sibling_indices)
    compute_dA_backward!(adj.proj.dA,
        adj.proj.phi, adj.proj.dphi, adj.proj.child_indices)
    reshape_explicit!(g, adj.proj.dA)
    return g
end


@inline (adj::RKOCProjectionAdjoint{T})(x::Vector{T}) where {T} =
    adj(similar(x), x)


end # module RungeKuttaToolKit
