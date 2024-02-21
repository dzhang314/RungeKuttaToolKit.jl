module RungeKuttaToolKit


using MultiFloats: MultiFloat, rsqrt


include("ButcherInstructions.jl")
using .ButcherInstructions: LevelSequence,
    ButcherInstruction, ButcherInstructionTable,
    all_rooted_trees, butcher_density, butcher_symmetry


###################################################### EVALUATOR DATA STRUCTURES


export RKOCEvaluatorAE, RKOCEvaluatorAI, RKOCEvaluatorBE, RKOCEvaluatorBI


struct RKOCEvaluatorAE{T}
    table::ButcherInstructionTable
    A::Matrix{T}
    dA::Matrix{T}
    phi::Matrix{T}
    dphi::Matrix{T}
    Q::Matrix{T}
    R::Matrix{T}
    b::Vector{T}
    inv_gamma::Vector{T}
    residuals::Vector{T}
end


struct RKOCEvaluatorAI{T}
    table::ButcherInstructionTable
    A::Matrix{T}
    dA::Matrix{T}
    phi::Matrix{T}
    dphi::Matrix{T}
    Q::Matrix{T}
    R::Matrix{T}
    b::Vector{T}
    inv_gamma::Vector{T}
    residuals::Vector{T}
end


struct RKOCEvaluatorBE{T}
    table::ButcherInstructionTable
    A::Matrix{T}
    dA::Matrix{T}
    b::Vector{T}
    db::Vector{T}
    phi::Matrix{T}
    dphi::Matrix{T}
    inv_gamma::Vector{T}
    residuals::Vector{T}
end


struct RKOCEvaluatorBI{T}
    table::ButcherInstructionTable
    A::Matrix{T}
    dA::Matrix{T}
    b::Vector{T}
    db::Vector{T}
    phi::Matrix{T}
    dphi::Matrix{T}
    inv_gamma::Vector{T}
    residuals::Vector{T}
end


function RKOCEvaluatorAE{T}(trees::Vector{LevelSequence}, s::Int) where {T}
    table = ButcherInstructionTable(trees)
    return RKOCEvaluatorAE{T}(table,
        Matrix{T}(undef, s, s), Matrix{T}(undef, s, s),
        Matrix{T}(undef, s, length(table.instructions)),
        Matrix{T}(undef, s, length(table.instructions)),
        Matrix{T}(undef, length(trees), s),
        Matrix{T}(undef, s, s), Vector{T}(undef, s),
        [inv(T(butcher_density(tree))) for tree in trees],
        Vector{T}(undef, length(trees)))
end


function RKOCEvaluatorAI{T}(trees::Vector{LevelSequence}, s::Int) where {T}
    table = ButcherInstructionTable(trees)
    return RKOCEvaluatorAI{T}(table,
        Matrix{T}(undef, s, s), Matrix{T}(undef, s, s),
        Matrix{T}(undef, s, length(table.instructions)),
        Matrix{T}(undef, s, length(table.instructions)),
        Matrix{T}(undef, length(trees), s),
        Matrix{T}(undef, s, s), Vector{T}(undef, s),
        [inv(T(butcher_density(tree))) for tree in trees],
        Vector{T}(undef, length(trees)))
end


function RKOCEvaluatorBE{T}(trees::Vector{LevelSequence}, s::Int) where {T}
    table = ButcherInstructionTable(trees)
    return RKOCEvaluatorBE{T}(table,
        Matrix{T}(undef, s, s), Matrix{T}(undef, s, s),
        Vector{T}(undef, s), Vector{T}(undef, s),
        Matrix{T}(undef, s, length(table.instructions)),
        Matrix{T}(undef, s, length(table.instructions)),
        [inv(T(butcher_density(tree))) for tree in trees],
        Vector{T}(undef, length(trees)))
end


function RKOCEvaluatorBI{T}(trees::Vector{LevelSequence}, s::Int) where {T}
    table = ButcherInstructionTable(trees)
    return RKOCEvaluatorBI{T}(table,
        Matrix{T}(undef, s, s), Matrix{T}(undef, s, s),
        Vector{T}(undef, s), Vector{T}(undef, s),
        Matrix{T}(undef, s, length(table.instructions)),
        Matrix{T}(undef, s, length(table.instructions)),
        [inv(T(butcher_density(tree))) for tree in trees],
        Vector{T}(undef, length(trees)))
end


@inline RKOCEvaluatorAE{T}(p::Int, s::Int) where {T} =
    RKOCEvaluatorAE{T}(all_rooted_trees(p), s)
@inline RKOCEvaluatorAI{T}(p::Int, s::Int) where {T} =
    RKOCEvaluatorAI{T}(all_rooted_trees(p), s)
@inline RKOCEvaluatorBE{T}(p::Int, s::Int) where {T} =
    RKOCEvaluatorBE{T}(all_rooted_trees(p), s)
@inline RKOCEvaluatorBI{T}(p::Int, s::Int) where {T} =
    RKOCEvaluatorBI{T}(all_rooted_trees(p), s)


#################################### PHI AND RESIDUAL COMPUTATION (FORWARD PASS)


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


function reshape_explicit!(
    A::AbstractMatrix{T}, x::AbstractVector{T}
) where {T}
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


function reshape_explicit!(
    x::AbstractVector{T}, A::AbstractMatrix{T}
) where {T}
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


function reshape_explicit!(
    A::AbstractMatrix{T}, b::AbstractVector{T}, x::AbstractVector{T}
) where {T}
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


function reshape_explicit!(
    x::AbstractVector{T}, A::AbstractMatrix{T}, b::AbstractVector{T}
) where {T}
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


function reshape_implicit!(
    A::AbstractMatrix{T}, x::AbstractVector{T}
) where {T}
    s = size(A, 1)
    @assert s == size(A, 2)
    @assert (s * s,) == size(x)
    offset = 0
    for i = 1:s
        @simd ivdep for j = 1:s
            @inbounds A[i, j] = x[offset+j]
        end
        offset += s
    end
    return A
end


function reshape_implicit!(
    x::AbstractVector{T}, A::AbstractMatrix{T}
) where {T}
    s = size(A, 1)
    @assert s == size(A, 2)
    @assert (s * s,) == size(x)
    offset = 0
    for i = 1:s
        @simd ivdep for j = 1:s
            @inbounds x[offset+j] = A[i, j]
        end
        offset += s
    end
    return x
end


function reshape_implicit!(
    A::AbstractMatrix{T}, b::AbstractVector{T}, x::AbstractVector{T}
) where {T}
    s = size(A, 1)
    @assert s == size(A, 2)
    @assert (s * (s + 1),) == size(x)
    offset = 0
    for i = 1:s
        @simd ivdep for j = 1:s
            @inbounds A[i, j] = x[offset+j]
        end
        offset += s
    end
    @simd ivdep for i = 1:s
        @inbounds b[i] = x[offset+i]
    end
    return (A, b)
end


function reshape_implicit!(
    x::AbstractVector{T}, A::AbstractMatrix{T}, b::AbstractVector{T}
) where {T}
    s = size(A, 1)
    @assert s == size(A, 2)
    @assert (s * (s + 1),) == size(x)
    offset = 0
    for i = 1:s
        @simd ivdep for j = 1:s
            @inbounds x[offset+j] = A[i, j]
        end
        offset += s
    end
    @simd ivdep for i = 1:s
        @inbounds x[offset+i] = b[i]
    end
    return x
end


function test_reshape_operators()
    let
        A, b = reshape_explicit!(
            Matrix{BigFloat}(undef, 3, 3),
            Vector{BigFloat}(undef, 3),
            BigFloat[1, 2, 3, 4, 5, 6])
        @assert A == BigFloat[0 0 0; 1 0 0; 2 3 0]
        @assert b == BigFloat[4, 5, 6]
    end
    let
        x = reshape_explicit!(
            Vector{BigFloat}(undef, 6),
            BigFloat[0 0 0; 1 0 0; 2 3 0],
            BigFloat[4, 5, 6])
        @assert x == BigFloat[1, 2, 3, 4, 5, 6]
    end
    let
        A = reshape_explicit!(
            Matrix{BigFloat}(undef, 3, 3),
            BigFloat[1, 2, 3])
        @assert A == BigFloat[0 0 0; 1 0 0; 2 3 0]
    end
    let
        x = reshape_explicit!(
            Vector{BigFloat}(undef, 3),
            BigFloat[0 0 0; 1 0 0; 2 3 0])
        @assert x == BigFloat[1, 2, 3]
    end
    let
        A, b = reshape_implicit!(
            Matrix{BigFloat}(undef, 3, 3),
            Vector{BigFloat}(undef, 3),
            BigFloat[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
        @assert A == BigFloat[1 2 3; 4 5 6; 7 8 9]
        @assert b == BigFloat[10, 11, 12]
    end
    let
        x = reshape_implicit!(
            Vector{BigFloat}(undef, 12),
            BigFloat[1 2 3; 4 5 6; 7 8 9],
            BigFloat[10, 11, 12])
        @assert x == BigFloat[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    end
    let
        A = reshape_implicit!(
            Matrix{BigFloat}(undef, 3, 3),
            BigFloat[1, 2, 3, 4, 5, 6, 7, 8, 9])
        @assert A == BigFloat[1 2 3; 4 5 6; 7 8 9]
    end
    let
        x = reshape_implicit!(
            Vector{BigFloat}(undef, 9),
            BigFloat[1 2 3; 4 5 6; 7 8 9])
        @assert x == BigFloat[1, 2, 3, 4, 5, 6, 7, 8, 9]
    end
    return true
end


############################################################## FUNCTOR INTERFACE


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


function residual_norm_squared(residuals::AbstractVector{T}) where {T}
    result = zero(T)
    for r in residuals
        result += abs2(r)
    end
    return result
end


function (ev::RKOCEvaluatorAE{T})(x::Vector{T}) where {T}
    reshape_explicit!(ev.A, x)
    compute_phi!(ev.phi, ev.A, ev.table.instructions)
    populate_Q!(ev.Q, ev.phi, ev.table.output_indices)
    gram_schmidt_qr!(ev.Q)
    compute_residuals!(ev.residuals, ev.Q, ev.inv_gamma)
    return residual_norm_squared(ev.residuals)
end


function (ev::RKOCEvaluatorAI{T})(x::Vector{T}) where {T}
    reshape_implicit!(ev.A, x)
    compute_phi!(ev.phi, ev.A, ev.table.instructions)
    populate_Q!(ev.Q, ev.phi, ev.table.output_indices)
    gram_schmidt_qr!(ev.Q)
    compute_residuals!(ev.residuals, ev.Q, ev.inv_gamma)
    return residual_norm_squared(ev.residuals)
end


function (ev::RKOCEvaluatorBE{T})(x::Vector{T}) where {T}
    reshape_explicit!(ev.A, ev.b, x)
    compute_phi!(ev.phi, ev.A, ev.table.instructions)
    compute_residuals!(ev.residuals,
        ev.b, ev.phi, ev.inv_gamma, ev.table.output_indices)
    return residual_norm_squared(ev.residuals)
end


function (ev::RKOCEvaluatorBI{T})(x::Vector{T}) where {T}
    reshape_implicit!(ev.A, ev.b, x)
    compute_phi!(ev.phi, ev.A, ev.table.instructions)
    compute_residuals!(ev.residuals,
        ev.b, ev.phi, ev.inv_gamma, ev.table.output_indices)
    return residual_norm_squared(ev.residuals)
end


struct RKOCEvaluatorAEAdjoint{T}
    ev::RKOCEvaluatorAE{T}
end
struct RKOCEvaluatorAIAdjoint{T}
    ev::RKOCEvaluatorAI{T}
end
struct RKOCEvaluatorBEAdjoint{T}
    ev::RKOCEvaluatorBE{T}
end
struct RKOCEvaluatorBIAdjoint{T}
    ev::RKOCEvaluatorBI{T}
end


@inline Base.adjoint(ev::RKOCEvaluatorAE{T}) where {T} =
    RKOCEvaluatorAEAdjoint{T}(ev)
@inline Base.adjoint(ev::RKOCEvaluatorAI{T}) where {T} =
    RKOCEvaluatorAIAdjoint{T}(ev)
@inline Base.adjoint(ev::RKOCEvaluatorBE{T}) where {T} =
    RKOCEvaluatorBEAdjoint{T}(ev)
@inline Base.adjoint(ev::RKOCEvaluatorBI{T}) where {T} =
    RKOCEvaluatorBIAdjoint{T}(ev)


function (adj::RKOCEvaluatorAEAdjoint{T})(g::Vector{T}, x::Vector{T}) where {T}
    reshape_explicit!(adj.ev.A, x)
    compute_phi!(adj.ev.phi, adj.ev.A, adj.ev.table.instructions)
    populate_Q!(adj.ev.Q, adj.ev.phi, adj.ev.table.output_indices)
    gram_schmidt_qr!(adj.ev.Q, adj.ev.R)
    compute_residuals!(adj.ev.residuals, adj.ev.b,
        adj.ev.Q, adj.ev.inv_gamma)
    solve_upper_triangular!(adj.ev.b, adj.ev.R)
    initialize_dphi_from_residual!(adj.ev.dphi,
        adj.ev.b, adj.ev.residuals, adj.ev.table.source_indices)
    compute_dphi_backward!(adj.ev.dphi,
        adj.ev.A, adj.ev.phi, adj.ev.table.child_indices,
        adj.ev.table.sibling_ranges, adj.ev.table.sibling_indices)
    compute_dA_backward!(adj.ev.dA,
        adj.ev.phi, adj.ev.dphi, adj.ev.table.child_indices)
    reshape_explicit!(g, adj.ev.dA)
    return g
end


function (adj::RKOCEvaluatorAIAdjoint{T})(g::Vector{T}, x::Vector{T}) where {T}
    reshape_implicit!(adj.ev.A, x)
    compute_phi!(adj.ev.phi, adj.ev.A, adj.ev.table.instructions)
    populate_Q!(adj.ev.Q, adj.ev.phi, adj.ev.table.output_indices)
    gram_schmidt_qr!(adj.ev.Q, adj.ev.R)
    compute_residuals!(adj.ev.residuals, adj.ev.b,
        adj.ev.Q, adj.ev.inv_gamma)
    solve_upper_triangular!(adj.ev.b, adj.ev.R)
    initialize_dphi_from_residual!(adj.ev.dphi,
        adj.ev.b, adj.ev.residuals, adj.ev.table.source_indices)
    compute_dphi_backward!(adj.ev.dphi,
        adj.ev.A, adj.ev.phi, adj.ev.table.child_indices,
        adj.ev.table.sibling_ranges, adj.ev.table.sibling_indices)
    compute_dA_backward!(adj.ev.dA,
        adj.ev.phi, adj.ev.dphi, adj.ev.table.child_indices)
    reshape_implicit!(g, adj.ev.dA)
    return g
end


function (adj::RKOCEvaluatorBEAdjoint{T})(g::Vector{T}, x::Vector{T}) where {T}
    reshape_explicit!(adj.ev.A, adj.ev.b, x)
    compute_phi!(adj.ev.phi, adj.ev.A, adj.ev.table.instructions)
    compute_residuals!(adj.ev.residuals,
        adj.ev.b, adj.ev.phi, adj.ev.inv_gamma, adj.ev.table.output_indices)
    initialize_dphi_from_residual!(adj.ev.dphi,
        adj.ev.b, adj.ev.residuals, adj.ev.table.source_indices)
    compute_dphi_backward!(adj.ev.dphi,
        adj.ev.A, adj.ev.phi, adj.ev.table.child_indices,
        adj.ev.table.sibling_ranges, adj.ev.table.sibling_indices)
    compute_dA_backward!(adj.ev.dA,
        adj.ev.phi, adj.ev.dphi, adj.ev.table.child_indices)
    compute_db_backward!(adj.ev.db,
        adj.ev.phi, adj.ev.residuals, adj.ev.table.output_indices)
    reshape_explicit!(g, adj.ev.dA, adj.ev.db)
    return g
end


function (adj::RKOCEvaluatorBIAdjoint{T})(g::Vector{T}, x::Vector{T}) where {T}
    reshape_implicit!(adj.ev.A, adj.ev.b, x)
    compute_phi!(adj.ev.phi, adj.ev.A, adj.ev.table.instructions)
    compute_residuals!(adj.ev.residuals,
        adj.ev.b, adj.ev.phi, adj.ev.inv_gamma, adj.ev.table.output_indices)
    initialize_dphi_from_residual!(adj.ev.dphi,
        adj.ev.b, adj.ev.residuals, adj.ev.table.source_indices)
    compute_dphi_backward!(adj.ev.dphi,
        adj.ev.A, adj.ev.phi, adj.ev.table.child_indices,
        adj.ev.table.sibling_ranges, adj.ev.table.sibling_indices)
    compute_dA_backward!(adj.ev.dA,
        adj.ev.phi, adj.ev.dphi, adj.ev.table.child_indices)
    compute_db_backward!(adj.ev.db,
        adj.ev.phi, adj.ev.residuals, adj.ev.table.output_indices)
    reshape_implicit!(g, adj.ev.dA, adj.ev.db)
    return g
end


@inline (adj::RKOCEvaluatorAEAdjoint{T})(x::Vector{T}) where {T} =
    adj(similar(x), x)
@inline (adj::RKOCEvaluatorAIAdjoint{T})(x::Vector{T}) where {T} =
    adj(similar(x), x)
@inline (adj::RKOCEvaluatorBEAdjoint{T})(x::Vector{T}) where {T} =
    adj(similar(x), x)
@inline (adj::RKOCEvaluatorBIAdjoint{T})(x::Vector{T}) where {T} =
    adj(similar(x), x)


end # module RungeKuttaToolKit
