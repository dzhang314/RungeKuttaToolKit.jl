module RungeKuttaToolKit


using MultiFloats: MultiFloat, rsqrt


include("ButcherInstructions.jl")
using .ButcherInstructions: LevelSequence,
    ButcherInstruction, ButcherInstructionTable,
    rooted_trees, all_rooted_trees, butcher_density, butcher_symmetry
export rooted_trees, all_rooted_trees, butcher_density, butcher_symmetry


###################################################### EVALUATOR DATA STRUCTURES


export RKOCEvaluatorAE, RKOCEvaluatorAD, RKOCEvaluatorAI
export RKOCEvaluatorBE, RKOCEvaluatorBD, RKOCEvaluatorBI
export RKOCResidualEvaluatorAE, RKOCResidualEvaluatorAD, RKOCResidualEvaluatorAI
export RKOCResidualEvaluatorBE, RKOCResidualEvaluatorBD, RKOCResidualEvaluatorBI


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


struct RKOCEvaluatorAD{T}
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


struct RKOCEvaluatorBD{T}
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


struct RKOCResidualEvaluatorAE{T}
    table::ButcherInstructionTable
    A::Matrix{T}
    b::Vector{T}
    db::Vector{T}
    phi::Matrix{T}
    dphi::Matrix{T}
    Q::Matrix{T}
    R::Matrix{T}
    inv_gamma::Vector{T}
    residuals::Vector{T}
end


struct RKOCResidualEvaluatorAD{T}
    table::ButcherInstructionTable
    A::Matrix{T}
    b::Vector{T}
    db::Vector{T}
    phi::Matrix{T}
    dphi::Matrix{T}
    Q::Matrix{T}
    R::Matrix{T}
    inv_gamma::Vector{T}
    residuals::Vector{T}
end


struct RKOCResidualEvaluatorAI{T}
    table::ButcherInstructionTable
    A::Matrix{T}
    b::Vector{T}
    db::Vector{T}
    phi::Matrix{T}
    dphi::Matrix{T}
    Q::Matrix{T}
    R::Matrix{T}
    inv_gamma::Vector{T}
    residuals::Vector{T}
end


struct RKOCResidualEvaluatorBE{T}
    table::ButcherInstructionTable
    A::Matrix{T}
    b::Vector{T}
    phi::Matrix{T}
    dphi::Matrix{T}
    inv_gamma::Vector{T}
end


struct RKOCResidualEvaluatorBD{T}
    table::ButcherInstructionTable
    A::Matrix{T}
    b::Vector{T}
    phi::Matrix{T}
    dphi::Matrix{T}
    inv_gamma::Vector{T}
end


struct RKOCResidualEvaluatorBI{T}
    table::ButcherInstructionTable
    A::Matrix{T}
    b::Vector{T}
    phi::Matrix{T}
    dphi::Matrix{T}
    inv_gamma::Vector{T}
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


function RKOCEvaluatorAD{T}(trees::Vector{LevelSequence}, s::Int) where {T}
    table = ButcherInstructionTable(trees)
    return RKOCEvaluatorAD{T}(table,
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


function RKOCEvaluatorBD{T}(trees::Vector{LevelSequence}, s::Int) where {T}
    table = ButcherInstructionTable(trees)
    return RKOCEvaluatorBD{T}(table,
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


function RKOCResidualEvaluatorAE{T}(
    trees::Vector{LevelSequence}, s::Int
) where {T}
    table = ButcherInstructionTable(trees)
    return RKOCResidualEvaluatorAE{T}(table,
        Matrix{T}(undef, s, s), Vector{T}(undef, s), Vector{T}(undef, s),
        Matrix{T}(undef, s, length(table.instructions)),
        Matrix{T}(undef, s, length(table.instructions)),
        Matrix{T}(undef, length(trees), s),
        Matrix{T}(undef, s, s),
        [inv(T(butcher_density(tree))) for tree in trees],
        Vector{T}(undef, length(trees)))
end


function RKOCResidualEvaluatorAD{T}(
    trees::Vector{LevelSequence}, s::Int
) where {T}
    table = ButcherInstructionTable(trees)
    return RKOCResidualEvaluatorAD{T}(table,
        Matrix{T}(undef, s, s), Vector{T}(undef, s), Vector{T}(undef, s),
        Matrix{T}(undef, s, length(table.instructions)),
        Matrix{T}(undef, s, length(table.instructions)),
        Matrix{T}(undef, length(trees), s),
        Matrix{T}(undef, s, s),
        [inv(T(butcher_density(tree))) for tree in trees],
        Vector{T}(undef, length(trees)))
end


function RKOCResidualEvaluatorAI{T}(
    trees::Vector{LevelSequence}, s::Int
) where {T}
    table = ButcherInstructionTable(trees)
    return RKOCResidualEvaluatorAI{T}(table,
        Matrix{T}(undef, s, s), Vector{T}(undef, s), Vector{T}(undef, s),
        Matrix{T}(undef, s, length(table.instructions)),
        Matrix{T}(undef, s, length(table.instructions)),
        Matrix{T}(undef, length(trees), s),
        Matrix{T}(undef, s, s),
        [inv(T(butcher_density(tree))) for tree in trees],
        Vector{T}(undef, length(trees)))
end


function RKOCResidualEvaluatorBE{T}(
    trees::Vector{LevelSequence}, s::Int
) where {T}
    table = ButcherInstructionTable(trees)
    return RKOCResidualEvaluatorBE{T}(table,
        Matrix{T}(undef, s, s), Vector{T}(undef, s),
        Matrix{T}(undef, s, length(table.instructions)),
        Matrix{T}(undef, s, length(table.instructions)),
        [inv(T(butcher_density(tree))) for tree in trees])
end


function RKOCResidualEvaluatorBD{T}(
    trees::Vector{LevelSequence}, s::Int
) where {T}
    table = ButcherInstructionTable(trees)
    return RKOCResidualEvaluatorBD{T}(table,
        Matrix{T}(undef, s, s), Vector{T}(undef, s),
        Matrix{T}(undef, s, length(table.instructions)),
        Matrix{T}(undef, s, length(table.instructions)),
        [inv(T(butcher_density(tree))) for tree in trees])
end


function RKOCResidualEvaluatorBI{T}(
    trees::Vector{LevelSequence}, s::Int
) where {T}
    table = ButcherInstructionTable(trees)
    return RKOCResidualEvaluatorBI{T}(table,
        Matrix{T}(undef, s, s), Vector{T}(undef, s),
        Matrix{T}(undef, s, length(table.instructions)),
        Matrix{T}(undef, s, length(table.instructions)),
        [inv(T(butcher_density(tree))) for tree in trees])
end


@inline RKOCEvaluatorAE{T}(p::Int, s::Int) where {T} =
    RKOCEvaluatorAE{T}(all_rooted_trees(p), s)
@inline RKOCEvaluatorAD{T}(p::Int, s::Int) where {T} =
    RKOCEvaluatorAD{T}(all_rooted_trees(p), s)
@inline RKOCEvaluatorAI{T}(p::Int, s::Int) where {T} =
    RKOCEvaluatorAI{T}(all_rooted_trees(p), s)
@inline RKOCEvaluatorBE{T}(p::Int, s::Int) where {T} =
    RKOCEvaluatorBE{T}(all_rooted_trees(p), s)
@inline RKOCEvaluatorBD{T}(p::Int, s::Int) where {T} =
    RKOCEvaluatorBD{T}(all_rooted_trees(p), s)
@inline RKOCEvaluatorBI{T}(p::Int, s::Int) where {T} =
    RKOCEvaluatorBI{T}(all_rooted_trees(p), s)
@inline RKOCResidualEvaluatorAE{T}(p::Int, s::Int) where {T} =
    RKOCResidualEvaluatorAE{T}(all_rooted_trees(p), s)
@inline RKOCResidualEvaluatorAD{T}(p::Int, s::Int) where {T} =
    RKOCResidualEvaluatorAD{T}(all_rooted_trees(p), s)
@inline RKOCResidualEvaluatorAI{T}(p::Int, s::Int) where {T} =
    RKOCResidualEvaluatorAI{T}(all_rooted_trees(p), s)
@inline RKOCResidualEvaluatorBE{T}(p::Int, s::Int) where {T} =
    RKOCResidualEvaluatorBE{T}(all_rooted_trees(p), s)
@inline RKOCResidualEvaluatorBD{T}(p::Int, s::Int) where {T} =
    RKOCResidualEvaluatorBD{T}(all_rooted_trees(p), s)
@inline RKOCResidualEvaluatorBI{T}(p::Int, s::Int) where {T} =
    RKOCResidualEvaluatorBI{T}(all_rooted_trees(p), s)


################################################### PHI AND RESIDUAL COMPUTATION


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
        overlap = _zero
        for j = 1:s
            overlap += b[j] * phi[j, k]
        end
        residuals[i] = overlap - inv_gamma[i]
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


function compute_residuals_and_b!(
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


############################################ GRADIENT COMPUTATION (FORWARD-MODE)


function pushforward_dphi!(
    dphi::AbstractMatrix{T}, phi::AbstractMatrix{T},
    A::AbstractMatrix{T}, dA::AbstractMatrix{T},
    instructions::Vector{ButcherInstruction}
) where {T}
    @assert size(dphi) == size(phi)
    s = size(dphi, 1)
    @assert (s, s) == size(dA)
    @assert (s, s) == size(A)
    @assert size(dphi, 2) == length(instructions)
    _zero = zero(T)
    @inbounds for (k, instruction) in enumerate(instructions)
        p, q = instruction.left, instruction.right
        if p == -1
            @assert q == -1
            @simd ivdep for j = 1:s
                dphi[j, k] = _zero
            end
        elseif q == -1
            @simd ivdep for j = 1:s
                dphi[j, k] = _zero
            end
            for i = 1:s
                temp = phi[i, p]
                dtemp = dphi[i, p]
                @simd ivdep for j = 1:s
                    dphi[j, k] += dtemp * A[j, i] + temp * dA[j, i]
                end
            end
        else
            @simd ivdep for j = 1:s
                dphi[j, k] = dphi[j, p] * phi[j, q] + phi[j, p] * dphi[j, q]
            end
        end
    end
    return dphi
end


function pushforward_dphi!(
    dphi::AbstractMatrix{T}, phi::AbstractMatrix{T},
    A::AbstractMatrix{T}, u::Int, v::Int,
    instructions::Vector{ButcherInstruction}
) where {T}
    @assert size(dphi) == size(phi)
    s = size(dphi, 1)
    @assert (s, s) == size(A)
    @assert size(dphi, 2) == length(instructions)
    _zero = zero(T)
    @inbounds for (k, instruction) in enumerate(instructions)
        p, q = instruction.left, instruction.right
        if p == -1
            @assert q == -1
            @simd ivdep for j = 1:s
                dphi[j, k] = _zero
            end
        elseif q == -1
            @simd ivdep for j = 1:s
                dphi[j, k] = _zero
            end
            dphi[u, k] = phi[v, p]
            for i = 1:s
                dtemp = dphi[i, p]
                @simd ivdep for j = 1:s
                    dphi[j, k] += dtemp * A[j, i]
                end
            end
        else
            @simd ivdep for j = 1:s
                dphi[j, k] = dphi[j, p] * phi[j, q] + phi[j, p] * dphi[j, q]
            end
        end
    end
    return dphi
end


function pushforward_db!(
    db::AbstractVector{T}, temp::AbstractVector{T},
    dphi::AbstractMatrix{T}, b::AbstractVector{T},
    Q::AbstractMatrix{T}, R::AbstractMatrix{T},
    output_indices::AbstractVector{Int}
) where {T}
    s = length(db)
    t = length(temp)
    @assert s == size(dphi, 1)
    @assert (s,) == size(b)
    @assert (t, s) == size(Q)
    @assert (s, s) == size(R)
    @assert (t,) == size(output_indices)
    _zero = zero(T)
    @inbounds for (i, k) in enumerate(output_indices)
        overlap = _zero
        for j = 1:s
            overlap += b[j] * dphi[j, k]
        end
        temp[i] = overlap
    end
    @inbounds for i = 1:s
        overlap = _zero
        for j = 1:t
            overlap += Q[j, i] * temp[j]
        end
        db[i] = -overlap
    end
    solve_upper_triangular!(db, R)
    return db
end


function pushforward_db!(
    db::AbstractVector{T}, temp::AbstractVector{T},
    residuals::AbstractVector{T}, dphi::AbstractMatrix{T},
    b::AbstractVector{T}, Q::AbstractMatrix{T}, R::AbstractMatrix{T},
    output_indices::AbstractVector{Int}
) where {T}
    s = length(db)
    t = length(temp)
    @assert (t,) == size(residuals)
    @assert s == size(dphi, 1)
    @assert (s,) == size(b)
    @assert (t, s) == size(Q)
    @assert (s, s) == size(R)
    @assert (t,) == size(output_indices)
    _zero = zero(T)
    @simd ivdep for i = 1:s
        @inbounds db[i] = _zero
    end
    @inbounds for (i, k) in enumerate(output_indices)
        r = residuals[i]
        @simd ivdep for j = 1:s
            db[j] += r * dphi[j, k]
        end
    end
    solve_lower_triangular!(db, R)
    @inbounds for (i, k) in enumerate(output_indices)
        overlap = _zero
        for j = 1:s
            overlap += b[j] * dphi[j, k]
        end
        temp[i] = overlap
    end
    @inbounds for i = 1:s
        overlap = _zero
        for j = 1:t
            overlap += Q[j, i] * temp[j]
        end
        db[i] = -(db[i] + overlap)
    end
    solve_upper_triangular!(db, R)
    return db
end


function pushforward_dresiduals!(
    dresiduals::AbstractVector{T},
    b::AbstractVector{T}, dphi::AbstractMatrix{T},
    output_indices::AbstractVector{Int}
) where {T}
    t = length(dresiduals)
    s = length(b)
    @assert s == size(dphi, 1)
    @assert (t,) == size(output_indices)
    _zero = zero(T)
    @inbounds for (i, k) in enumerate(output_indices)
        doverlap = _zero
        for j = 1:s
            doverlap += b[j] * dphi[j, k]
        end
        dresiduals[i] = doverlap
    end
    return dresiduals
end


function pushforward_dresiduals!(
    dresiduals::AbstractVector{T},
    db::AbstractVector{T}, b::AbstractVector{T},
    dphi::AbstractMatrix{T}, phi::AbstractMatrix{T},
    output_indices::AbstractVector{Int}
) where {T}
    t = length(dresiduals)
    s = length(db)
    @assert (s,) == size(b)
    @assert s == size(dphi, 1)
    @assert size(dphi) == size(phi)
    @assert (t,) == size(output_indices)
    _zero = zero(T)
    @inbounds for (i, k) in enumerate(output_indices)
        doverlap = _zero
        for j = 1:s
            doverlap += db[j] * phi[j, k] + b[j] * dphi[j, k]
        end
        dresiduals[i] = doverlap
    end
    return dresiduals
end


############################################ GRADIENT COMPUTATION (REVERSE-MODE)


function pullback_dphi_from_residual!(
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


function pullback_dphi!(
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


function pullback_dA!(
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


function pullback_db!(
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
    # NOTE: R is stored transposed, and its diagonal is stored inverted.
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


function solve_lower_triangular!(
    b::AbstractVector{T}, L::AbstractMatrix{T}
) where {T}
    s = length(b)
    @assert (s, s) == size(L)
    # NOTE: The diagonal of L is stored inverted.
    _zero = zero(T)
    @inbounds for i = 1:s
        if iszero(L[i, i])
            b[i] = _zero
        else
            b[i] *= L[i, i]
            for j = i+1:s
                b[j] -= L[j, i] * b[i]
            end
        end
    end
    return b
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


function reshape_diagonally_implicit!(
    A::AbstractMatrix{T}, x::AbstractVector{T}
) where {T}
    s = size(A, 1)
    @assert s == size(A, 2)
    @assert ((s * (s + 1)) >> 1,) == size(x)
    _zero = zero(T)
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


function reshape_diagonally_implicit!(
    x::AbstractVector{T}, A::AbstractMatrix{T}
) where {T}
    s = size(A, 1)
    @assert s == size(A, 2)
    @assert ((s * (s + 1)) >> 1,) == size(x)
    offset = 0
    for i = 1:s
        @simd ivdep for j = 1:i
            @inbounds x[offset+j] = A[i, j]
        end
        offset += i
    end
    return x
end


function reshape_diagonally_implicit!(
    A::AbstractMatrix{T}, b::AbstractVector{T}, x::AbstractVector{T}
) where {T}
    s = length(b)
    @assert (s, s) == size(A)
    @assert ((s * (s + 3)) >> 1,) == size(x)
    _zero = zero(T)
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
    @simd ivdep for i = 1:s
        @inbounds b[i] = x[offset+i]
    end
    return (A, b)
end


function reshape_diagonally_implicit!(
    x::AbstractVector{T}, A::AbstractMatrix{T}, b::AbstractVector{T}
) where {T}
    s = length(b)
    @assert (s, s) == size(A)
    @assert ((s * (s + 3)) >> 1,) == size(x)
    offset = 0
    for i = 1:s
        @simd ivdep for j = 1:i
            @inbounds x[offset+j] = A[i, j]
        end
        offset += i
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


function (ev::RKOCEvaluatorAD{T})(x::Vector{T}) where {T}
    reshape_diagonally_implicit!(ev.A, x)
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


function (ev::RKOCEvaluatorBD{T})(x::Vector{T}) where {T}
    reshape_diagonally_implicit!(ev.A, ev.b, x)
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


function (ev::RKOCResidualEvaluatorAE{T})(
    residuals::Vector{T}, x::Vector{T}
) where {T}
    reshape_explicit!(ev.A, x)
    compute_phi!(ev.phi, ev.A, ev.table.instructions)
    populate_Q!(ev.Q, ev.phi, ev.table.output_indices)
    gram_schmidt_qr!(ev.Q)
    compute_residuals!(residuals, ev.Q, ev.inv_gamma)
    return residuals
end


function (ev::RKOCResidualEvaluatorAD{T})(
    residuals::Vector{T}, x::Vector{T}
) where {T}
    reshape_diagonally_implicit!(ev.A, x)
    compute_phi!(ev.phi, ev.A, ev.table.instructions)
    populate_Q!(ev.Q, ev.phi, ev.table.output_indices)
    gram_schmidt_qr!(ev.Q)
    compute_residuals!(residuals, ev.Q, ev.inv_gamma)
    return residuals
end


function (ev::RKOCResidualEvaluatorAI{T})(
    residuals::Vector{T}, x::Vector{T}
) where {T}
    reshape_implicit!(ev.A, x)
    compute_phi!(ev.phi, ev.A, ev.table.instructions)
    populate_Q!(ev.Q, ev.phi, ev.table.output_indices)
    gram_schmidt_qr!(ev.Q)
    compute_residuals!(residuals, ev.Q, ev.inv_gamma)
    return residuals
end


function (ev::RKOCResidualEvaluatorBE{T})(
    residuals::Vector{T}, x::Vector{T}
) where {T}
    reshape_explicit!(ev.A, ev.b, x)
    compute_phi!(ev.phi, ev.A, ev.table.instructions)
    compute_residuals!(residuals,
        ev.b, ev.phi, ev.inv_gamma, ev.table.output_indices)
    return residuals
end


function (ev::RKOCResidualEvaluatorBD{T})(
    residuals::Vector{T}, x::Vector{T}
) where {T}
    reshape_diagonally_implicit!(ev.A, ev.b, x)
    compute_phi!(ev.phi, ev.A, ev.table.instructions)
    compute_residuals!(residuals,
        ev.b, ev.phi, ev.inv_gamma, ev.table.output_indices)
    return residuals
end


function (ev::RKOCResidualEvaluatorBI{T})(
    residuals::Vector{T}, x::Vector{T}
) where {T}
    reshape_implicit!(ev.A, ev.b, x)
    compute_phi!(ev.phi, ev.A, ev.table.instructions)
    compute_residuals!(residuals,
        ev.b, ev.phi, ev.inv_gamma, ev.table.output_indices)
    return residuals
end


@inline (ev::RKOCResidualEvaluatorAE{T})(x::Vector{T}) where {T} =
    ev(Vector{T}(undef, length(ev.table.output_indices)), x)
@inline (ev::RKOCResidualEvaluatorAD{T})(x::Vector{T}) where {T} =
    ev(Vector{T}(undef, length(ev.table.output_indices)), x)
@inline (ev::RKOCResidualEvaluatorAI{T})(x::Vector{T}) where {T} =
    ev(Vector{T}(undef, length(ev.table.output_indices)), x)
@inline (ev::RKOCResidualEvaluatorBE{T})(x::Vector{T}) where {T} =
    ev(Vector{T}(undef, length(ev.table.output_indices)), x)
@inline (ev::RKOCResidualEvaluatorBD{T})(x::Vector{T}) where {T} =
    ev(Vector{T}(undef, length(ev.table.output_indices)), x)
@inline (ev::RKOCResidualEvaluatorBI{T})(x::Vector{T}) where {T} =
    ev(Vector{T}(undef, length(ev.table.output_indices)), x)


struct RKOCEvaluatorAEAdjoint{T}
    ev::RKOCEvaluatorAE{T}
end
struct RKOCEvaluatorADAdjoint{T}
    ev::RKOCEvaluatorAD{T}
end
struct RKOCEvaluatorAIAdjoint{T}
    ev::RKOCEvaluatorAI{T}
end
struct RKOCEvaluatorBEAdjoint{T}
    ev::RKOCEvaluatorBE{T}
end
struct RKOCEvaluatorBDAdjoint{T}
    ev::RKOCEvaluatorBD{T}
end
struct RKOCEvaluatorBIAdjoint{T}
    ev::RKOCEvaluatorBI{T}
end
struct RKOCResidualEvaluatorAEAdjoint{T}
    ev::RKOCResidualEvaluatorAE{T}
end
struct RKOCResidualEvaluatorADAdjoint{T}
    ev::RKOCResidualEvaluatorAD{T}
end
struct RKOCResidualEvaluatorAIAdjoint{T}
    ev::RKOCResidualEvaluatorAI{T}
end
struct RKOCResidualEvaluatorBEAdjoint{T}
    ev::RKOCResidualEvaluatorBE{T}
end
struct RKOCResidualEvaluatorBDAdjoint{T}
    ev::RKOCResidualEvaluatorBD{T}
end
struct RKOCResidualEvaluatorBIAdjoint{T}
    ev::RKOCResidualEvaluatorBI{T}
end


@inline Base.adjoint(ev::RKOCEvaluatorAE{T}) where {T} =
    RKOCEvaluatorAEAdjoint{T}(ev)
@inline Base.adjoint(ev::RKOCEvaluatorAD{T}) where {T} =
    RKOCEvaluatorADAdjoint{T}(ev)
@inline Base.adjoint(ev::RKOCEvaluatorAI{T}) where {T} =
    RKOCEvaluatorAIAdjoint{T}(ev)
@inline Base.adjoint(ev::RKOCEvaluatorBE{T}) where {T} =
    RKOCEvaluatorBEAdjoint{T}(ev)
@inline Base.adjoint(ev::RKOCEvaluatorBD{T}) where {T} =
    RKOCEvaluatorBDAdjoint{T}(ev)
@inline Base.adjoint(ev::RKOCEvaluatorBI{T}) where {T} =
    RKOCEvaluatorBIAdjoint{T}(ev)
@inline Base.adjoint(ev::RKOCResidualEvaluatorAE{T}) where {T} =
    RKOCResidualEvaluatorAEAdjoint{T}(ev)
@inline Base.adjoint(ev::RKOCResidualEvaluatorAD{T}) where {T} =
    RKOCResidualEvaluatorADAdjoint{T}(ev)
@inline Base.adjoint(ev::RKOCResidualEvaluatorAI{T}) where {T} =
    RKOCResidualEvaluatorAIAdjoint{T}(ev)
@inline Base.adjoint(ev::RKOCResidualEvaluatorBE{T}) where {T} =
    RKOCResidualEvaluatorBEAdjoint{T}(ev)
@inline Base.adjoint(ev::RKOCResidualEvaluatorBD{T}) where {T} =
    RKOCResidualEvaluatorBDAdjoint{T}(ev)
@inline Base.adjoint(ev::RKOCResidualEvaluatorBI{T}) where {T} =
    RKOCResidualEvaluatorBIAdjoint{T}(ev)


function (adj::RKOCEvaluatorAEAdjoint{T})(g::Vector{T}, x::Vector{T}) where {T}
    reshape_explicit!(adj.ev.A, x)
    compute_phi!(adj.ev.phi, adj.ev.A, adj.ev.table.instructions)
    populate_Q!(adj.ev.Q, adj.ev.phi, adj.ev.table.output_indices)
    gram_schmidt_qr!(adj.ev.Q, adj.ev.R)
    compute_residuals_and_b!(adj.ev.residuals, adj.ev.b,
        adj.ev.Q, adj.ev.inv_gamma)
    solve_upper_triangular!(adj.ev.b, adj.ev.R)
    pullback_dphi_from_residual!(adj.ev.dphi,
        adj.ev.b, adj.ev.residuals, adj.ev.table.source_indices)
    pullback_dphi!(adj.ev.dphi,
        adj.ev.A, adj.ev.phi, adj.ev.table.child_indices,
        adj.ev.table.sibling_ranges, adj.ev.table.sibling_indices)
    pullback_dA!(adj.ev.dA,
        adj.ev.phi, adj.ev.dphi, adj.ev.table.child_indices)
    reshape_explicit!(g, adj.ev.dA)
    return g
end


function (adj::RKOCEvaluatorADAdjoint{T})(g::Vector{T}, x::Vector{T}) where {T}
    reshape_diagonally_implicit!(adj.ev.A, x)
    compute_phi!(adj.ev.phi, adj.ev.A, adj.ev.table.instructions)
    populate_Q!(adj.ev.Q, adj.ev.phi, adj.ev.table.output_indices)
    gram_schmidt_qr!(adj.ev.Q, adj.ev.R)
    compute_residuals_and_b!(adj.ev.residuals, adj.ev.b,
        adj.ev.Q, adj.ev.inv_gamma)
    solve_upper_triangular!(adj.ev.b, adj.ev.R)
    pullback_dphi_from_residual!(adj.ev.dphi,
        adj.ev.b, adj.ev.residuals, adj.ev.table.source_indices)
    pullback_dphi!(adj.ev.dphi,
        adj.ev.A, adj.ev.phi, adj.ev.table.child_indices,
        adj.ev.table.sibling_ranges, adj.ev.table.sibling_indices)
    pullback_dA!(adj.ev.dA,
        adj.ev.phi, adj.ev.dphi, adj.ev.table.child_indices)
    reshape_diagonally_implicit!(g, adj.ev.dA)
    return g
end


function (adj::RKOCEvaluatorAIAdjoint{T})(g::Vector{T}, x::Vector{T}) where {T}
    reshape_implicit!(adj.ev.A, x)
    compute_phi!(adj.ev.phi, adj.ev.A, adj.ev.table.instructions)
    populate_Q!(adj.ev.Q, adj.ev.phi, adj.ev.table.output_indices)
    gram_schmidt_qr!(adj.ev.Q, adj.ev.R)
    compute_residuals_and_b!(adj.ev.residuals, adj.ev.b,
        adj.ev.Q, adj.ev.inv_gamma)
    solve_upper_triangular!(adj.ev.b, adj.ev.R)
    pullback_dphi_from_residual!(adj.ev.dphi,
        adj.ev.b, adj.ev.residuals, adj.ev.table.source_indices)
    pullback_dphi!(adj.ev.dphi,
        adj.ev.A, adj.ev.phi, adj.ev.table.child_indices,
        adj.ev.table.sibling_ranges, adj.ev.table.sibling_indices)
    pullback_dA!(adj.ev.dA,
        adj.ev.phi, adj.ev.dphi, adj.ev.table.child_indices)
    reshape_implicit!(g, adj.ev.dA)
    return g
end


function (adj::RKOCEvaluatorBEAdjoint{T})(g::Vector{T}, x::Vector{T}) where {T}
    reshape_explicit!(adj.ev.A, adj.ev.b, x)
    compute_phi!(adj.ev.phi, adj.ev.A, adj.ev.table.instructions)
    compute_residuals!(adj.ev.residuals,
        adj.ev.b, adj.ev.phi, adj.ev.inv_gamma, adj.ev.table.output_indices)
    pullback_dphi_from_residual!(adj.ev.dphi,
        adj.ev.b, adj.ev.residuals, adj.ev.table.source_indices)
    pullback_dphi!(adj.ev.dphi,
        adj.ev.A, adj.ev.phi, adj.ev.table.child_indices,
        adj.ev.table.sibling_ranges, adj.ev.table.sibling_indices)
    pullback_dA!(adj.ev.dA,
        adj.ev.phi, adj.ev.dphi, adj.ev.table.child_indices)
    pullback_db!(adj.ev.db,
        adj.ev.phi, adj.ev.residuals, adj.ev.table.output_indices)
    reshape_explicit!(g, adj.ev.dA, adj.ev.db)
    return g
end


function (adj::RKOCEvaluatorBDAdjoint{T})(g::Vector{T}, x::Vector{T}) where {T}
    reshape_diagonally_implicit!(adj.ev.A, adj.ev.b, x)
    compute_phi!(adj.ev.phi, adj.ev.A, adj.ev.table.instructions)
    compute_residuals!(adj.ev.residuals,
        adj.ev.b, adj.ev.phi, adj.ev.inv_gamma, adj.ev.table.output_indices)
    pullback_dphi_from_residual!(adj.ev.dphi,
        adj.ev.b, adj.ev.residuals, adj.ev.table.source_indices)
    pullback_dphi!(adj.ev.dphi,
        adj.ev.A, adj.ev.phi, adj.ev.table.child_indices,
        adj.ev.table.sibling_ranges, adj.ev.table.sibling_indices)
    pullback_dA!(adj.ev.dA,
        adj.ev.phi, adj.ev.dphi, adj.ev.table.child_indices)
    pullback_db!(adj.ev.db,
        adj.ev.phi, adj.ev.residuals, adj.ev.table.output_indices)
    reshape_diagonally_implicit!(g, adj.ev.dA, adj.ev.db)
    return g
end


function (adj::RKOCEvaluatorBIAdjoint{T})(g::Vector{T}, x::Vector{T}) where {T}
    reshape_implicit!(adj.ev.A, adj.ev.b, x)
    compute_phi!(adj.ev.phi, adj.ev.A, adj.ev.table.instructions)
    compute_residuals!(adj.ev.residuals,
        adj.ev.b, adj.ev.phi, adj.ev.inv_gamma, adj.ev.table.output_indices)
    pullback_dphi_from_residual!(adj.ev.dphi,
        adj.ev.b, adj.ev.residuals, adj.ev.table.source_indices)
    pullback_dphi!(adj.ev.dphi,
        adj.ev.A, adj.ev.phi, adj.ev.table.child_indices,
        adj.ev.table.sibling_ranges, adj.ev.table.sibling_indices)
    pullback_dA!(adj.ev.dA,
        adj.ev.phi, adj.ev.dphi, adj.ev.table.child_indices)
    pullback_db!(adj.ev.db,
        adj.ev.phi, adj.ev.residuals, adj.ev.table.output_indices)
    reshape_implicit!(g, adj.ev.dA, adj.ev.db)
    return g
end


function (adj::RKOCResidualEvaluatorAEAdjoint{T})(
    jacobian::Matrix{T}, x::Vector{T}
) where {T}
    @assert length(adj.ev.table.output_indices) == size(jacobian, 1)
    @assert length(x) == size(jacobian, 2)
    reshape_explicit!(adj.ev.A, x)
    compute_phi!(adj.ev.phi, adj.ev.A, adj.ev.table.instructions)
    populate_Q!(adj.ev.Q, adj.ev.phi, adj.ev.table.output_indices)
    gram_schmidt_qr!(adj.ev.Q, adj.ev.R)
    compute_residuals_and_b!(adj.ev.residuals, adj.ev.b,
        adj.ev.Q, adj.ev.inv_gamma)
    solve_upper_triangular!(adj.ev.b, adj.ev.R)
    s = length(adj.ev.b)
    k = 1
    for i = 2:s
        for j = 1:i-1
            pushforward_dphi!(adj.ev.dphi,
                adj.ev.phi, adj.ev.A, i, j, adj.ev.table.instructions)
            column = view(jacobian, :, k)
            pushforward_db!(adj.ev.db, column,
                adj.ev.residuals, adj.ev.dphi, adj.ev.b, adj.ev.Q, adj.ev.R,
                adj.ev.table.output_indices)
            pushforward_dresiduals!(column,
                adj.ev.db, adj.ev.b, adj.ev.dphi, adj.ev.phi,
                adj.ev.table.output_indices)
            k += 1
        end
    end
    return jacobian
end


function (adj::RKOCResidualEvaluatorADAdjoint{T})(
    jacobian::Matrix{T}, x::Vector{T}
) where {T}
    @assert length(adj.ev.table.output_indices) == size(jacobian, 1)
    @assert length(x) == size(jacobian, 2)
    reshape_diagonally_implicit!(adj.ev.A, x)
    compute_phi!(adj.ev.phi, adj.ev.A, adj.ev.table.instructions)
    populate_Q!(adj.ev.Q, adj.ev.phi, adj.ev.table.output_indices)
    gram_schmidt_qr!(adj.ev.Q, adj.ev.R)
    compute_residuals_and_b!(adj.ev.residuals, adj.ev.b,
        adj.ev.Q, adj.ev.inv_gamma)
    solve_upper_triangular!(adj.ev.b, adj.ev.R)
    s = length(adj.ev.b)
    k = 1
    for i = 1:s
        for j = 1:i
            pushforward_dphi!(adj.ev.dphi,
                adj.ev.phi, adj.ev.A, i, j, adj.ev.table.instructions)
            column = view(jacobian, :, k)
            pushforward_db!(adj.ev.db, column,
                adj.ev.residuals, adj.ev.dphi, adj.ev.b, adj.ev.Q, adj.ev.R,
                adj.ev.table.output_indices)
            pushforward_dresiduals!(column,
                adj.ev.db, adj.ev.b, adj.ev.dphi, adj.ev.phi,
                adj.ev.table.output_indices)
            k += 1
        end
    end
    return jacobian
end


function (adj::RKOCResidualEvaluatorAIAdjoint{T})(
    jacobian::Matrix{T}, x::Vector{T}
) where {T}
    @assert length(adj.ev.table.output_indices) == size(jacobian, 1)
    @assert length(x) == size(jacobian, 2)
    reshape_implicit!(adj.ev.A, x)
    compute_phi!(adj.ev.phi, adj.ev.A, adj.ev.table.instructions)
    populate_Q!(adj.ev.Q, adj.ev.phi, adj.ev.table.output_indices)
    gram_schmidt_qr!(adj.ev.Q, adj.ev.R)
    compute_residuals_and_b!(adj.ev.residuals, adj.ev.b,
        adj.ev.Q, adj.ev.inv_gamma)
    solve_upper_triangular!(adj.ev.b, adj.ev.R)
    s = length(adj.ev.b)
    k = 1
    for i = 1:s
        for j = 1:s
            pushforward_dphi!(adj.ev.dphi,
                adj.ev.phi, adj.ev.A, i, j, adj.ev.table.instructions)
            column = view(jacobian, :, k)
            pushforward_db!(adj.ev.db, column,
                adj.ev.residuals, adj.ev.dphi, adj.ev.b, adj.ev.Q, adj.ev.R,
                adj.ev.table.output_indices)
            pushforward_dresiduals!(column,
                adj.ev.db, adj.ev.b, adj.ev.dphi, adj.ev.phi,
                adj.ev.table.output_indices)
            k += 1
        end
    end
    return jacobian
end


function (adj::RKOCResidualEvaluatorBEAdjoint{T})(
    jacobian::Matrix{T}, x::Vector{T}
) where {T}
    @assert length(adj.ev.table.output_indices) == size(jacobian, 1)
    @assert length(x) == size(jacobian, 2)
    reshape_explicit!(adj.ev.A, adj.ev.b, x)
    compute_phi!(adj.ev.phi, adj.ev.A, adj.ev.table.instructions)
    s = length(adj.ev.b)
    k = 1
    for i = 2:s
        for j = 1:i-1
            pushforward_dphi!(adj.ev.dphi,
                adj.ev.phi, adj.ev.A, i, j, adj.ev.table.instructions)
            pushforward_dresiduals!(view(jacobian, :, k),
                adj.ev.b, adj.ev.dphi, adj.ev.table.output_indices)
            k += 1
        end
    end
    for j = 1:s
        for (i, m) in enumerate(adj.ev.table.output_indices)
            @inbounds jacobian[i, k] = adj.ev.phi[j, m]
        end
        k += 1
    end
    return jacobian
end


function (adj::RKOCResidualEvaluatorBDAdjoint{T})(
    jacobian::Matrix{T}, x::Vector{T}
) where {T}
    @assert length(adj.ev.table.output_indices) == size(jacobian, 1)
    @assert length(x) == size(jacobian, 2)
    reshape_diagonally_implicit!(adj.ev.A, adj.ev.b, x)
    compute_phi!(adj.ev.phi, adj.ev.A, adj.ev.table.instructions)
    s = length(adj.ev.b)
    k = 1
    for i = 1:s
        for j = 1:i
            pushforward_dphi!(adj.ev.dphi,
                adj.ev.phi, adj.ev.A, i, j, adj.ev.table.instructions)
            pushforward_dresiduals!(view(jacobian, :, k),
                adj.ev.b, adj.ev.dphi, adj.ev.table.output_indices)
            k += 1
        end
    end
    for j = 1:s
        for (i, m) in enumerate(adj.ev.table.output_indices)
            @inbounds jacobian[i, k] = adj.ev.phi[j, m]
        end
        k += 1
    end
    return jacobian
end


function (adj::RKOCResidualEvaluatorBIAdjoint{T})(
    jacobian::Matrix{T}, x::Vector{T}
) where {T}
    @assert length(adj.ev.table.output_indices) == size(jacobian, 1)
    @assert length(x) == size(jacobian, 2)
    reshape_implicit!(adj.ev.A, adj.ev.b, x)
    compute_phi!(adj.ev.phi, adj.ev.A, adj.ev.table.instructions)
    s = length(adj.ev.b)
    k = 1
    for i = 1:s
        for j = 1:s
            pushforward_dphi!(adj.ev.dphi,
                adj.ev.phi, adj.ev.A, i, j, adj.ev.table.instructions)
            pushforward_dresiduals!(view(jacobian, :, k),
                adj.ev.b, adj.ev.dphi, adj.ev.table.output_indices)
            k += 1
        end
    end
    for j = 1:s
        for (i, m) in enumerate(adj.ev.table.output_indices)
            @inbounds jacobian[i, k] = adj.ev.phi[j, m]
        end
        k += 1
    end
    return jacobian
end


@inline (adj::RKOCEvaluatorAEAdjoint{T})(x::Vector{T}) where {T} =
    adj(similar(x), x)
@inline (adj::RKOCEvaluatorADAdjoint{T})(x::Vector{T}) where {T} =
    adj(similar(x), x)
@inline (adj::RKOCEvaluatorAIAdjoint{T})(x::Vector{T}) where {T} =
    adj(similar(x), x)
@inline (adj::RKOCEvaluatorBEAdjoint{T})(x::Vector{T}) where {T} =
    adj(similar(x), x)
@inline (adj::RKOCEvaluatorBDAdjoint{T})(x::Vector{T}) where {T} =
    adj(similar(x), x)
@inline (adj::RKOCEvaluatorBIAdjoint{T})(x::Vector{T}) where {T} =
    adj(similar(x), x)
@inline (adj::RKOCResidualEvaluatorAEAdjoint{T})(x::Vector{T}) where {T} =
    adj(Matrix{T}(undef, length(adj.ev.table.output_indices), length(x)), x)
@inline (adj::RKOCResidualEvaluatorADAdjoint{T})(x::Vector{T}) where {T} =
    adj(Matrix{T}(undef, length(adj.ev.table.output_indices), length(x)), x)
@inline (adj::RKOCResidualEvaluatorAIAdjoint{T})(x::Vector{T}) where {T} =
    adj(Matrix{T}(undef, length(adj.ev.table.output_indices), length(x)), x)
@inline (adj::RKOCResidualEvaluatorBEAdjoint{T})(x::Vector{T}) where {T} =
    adj(Matrix{T}(undef, length(adj.ev.table.output_indices), length(x)), x)
@inline (adj::RKOCResidualEvaluatorBDAdjoint{T})(x::Vector{T}) where {T} =
    adj(Matrix{T}(undef, length(adj.ev.table.output_indices), length(x)), x)
@inline (adj::RKOCResidualEvaluatorBIAdjoint{T})(x::Vector{T}) where {T} =
    adj(Matrix{T}(undef, length(adj.ev.table.output_indices), length(x)), x)


end # module RungeKuttaToolKit
