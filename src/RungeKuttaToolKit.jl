module RungeKuttaToolKit


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


RKOCEvaluator{T}(p::Int, s::Int) where {T} =
    RKOCEvaluator{T}(all_rooted_trees(p), s)


############################################ RESIDUAL COMPUTATION (FORWARD PASS)


function compute_phi!(
    phi::AbstractMatrix{T}, A::AbstractMatrix{T},
    instructions::Vector{ButcherInstruction}
) where {T}
    n = size(phi, 1)
    @assert (n, n) == size(A)
    @assert size(phi, 2) == length(instructions)
    _zero = zero(T)
    _one = one(T)
    @inbounds for (k, instruction) in enumerate(instructions)
        p, q = instruction.left, instruction.right
        if p == -1
            @assert q == -1
            @simd ivdep for j = 1:n
                phi[j, k] = _one
            end
        elseif q == -1
            @simd ivdep for j = 1:n
                phi[j, k] = _zero
            end
            for i = 1:n
                temp = phi[i, p]
                @simd ivdep for j = 1:n
                    phi[j, k] += temp * A[j, i]
                end
            end
        else
            @simd ivdep for j = 1:n
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
    m = length(residuals)
    n = length(b)
    @assert n == size(phi, 1)
    @assert (m,) == size(inv_gamma)
    @assert (m,) == size(output_indices)
    _zero = zero(T)
    @inbounds for (i, k) in enumerate(output_indices)
        result = _zero
        for j = 1:n
            result += b[j] * phi[j, k]
        end
        residuals[i] = result - inv_gamma[i]
    end
    return nothing
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
    n = size(A, 1)
    @assert n == size(A, 2)
    @assert ((n * (n - 1)) >> 1,) == size(x)
    _zero = zero(T)
    offset = 0
    for i = 1:n
        @simd ivdep for j = 1:i-1
            @inbounds A[i, j] = x[offset+j]
        end
        offset += i - 1
        @simd ivdep for j = i:n
            @inbounds A[i, j] = _zero
        end
    end
    return nothing
end


function reshape_explicit!(x::Vector{T}, A::Matrix{T}) where {T}
    n = size(A, 1)
    @assert n == size(A, 2)
    @assert ((n * (n - 1)) >> 1,) == size(x)
    offset = 0
    for i = 2:n
        @simd ivdep for j = 1:i-1
            @inbounds x[offset+j] = A[i, j]
        end
        offset += i - 1
    end
    return nothing
end


function reshape_explicit!(A::Matrix{T}, b::Vector{T}, x::Vector{T}) where {T}
    n = length(b)
    @assert (n, n) == size(A)
    @assert ((n * (n + 1)) >> 1,) == size(x)
    _zero = zero(T)
    offset = 0
    for i = 1:n
        @simd ivdep for j = 1:i-1
            @inbounds A[i, j] = x[offset+j]
        end
        offset += i - 1
        @simd ivdep for j = i:n
            @inbounds A[i, j] = _zero
        end
    end
    @simd ivdep for i = 1:n
        @inbounds b[i] = x[offset+i]
    end
    return nothing
end


function reshape_explicit!(x::Vector{T}, A::Matrix{T}, b::Vector{T}) where {T}
    n = length(b)
    @assert (n, n) == size(A)
    @assert ((n * (n + 1)) >> 1,) == size(x)
    offset = 0
    for i = 2:n
        @simd ivdep for j = 1:i-1
            @inbounds x[offset+j] = A[i, j]
        end
        offset += i - 1
    end
    @simd ivdep for i = 1:n
        @inbounds x[offset+i] = b[i]
    end
    return nothing
end


function reshape_implicit!(A::Matrix{T}, b::Vector{T}, x::Vector{T}) where {T}
    n = length(b)
    @assert (n, n) == size(A)
    @assert (n * (n + 1),) == size(x)
    n_squared = n * n
    @simd ivdep for i = 1:n_squared
        @inbounds A[i] = x[i]
    end
    @simd ivdep for i = 1:n
        @inbounds b[i] = x[n_squared+i]
    end
    return nothing
end


function reshape_implicit!(x::Vector{T}, A::Matrix{T}, b::Vector{T}) where {T}
    n = length(b)
    @assert (n, n) == size(A)
    @assert (n * (n + 1),) == size(x)
    n_squared = n * n
    @simd ivdep for i = 1:n_squared
        @inbounds x[i] = A[i]
    end
    @simd ivdep for i = 1:n
        @inbounds x[n_squared+i] = b[i]
    end
    return nothing
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


struct RKOCEvaluatorAdjoint{T}
    ev::RKOCEvaluator{T}
end


Base.adjoint(ev::RKOCEvaluator{T}) where {T} = RKOCEvaluatorAdjoint{T}(ev)


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


(adj::RKOCEvaluatorAdjoint{T})(x::Vector{T}) where {T} = adj(similar(x), x)


#=

using MultiFloats

export RKOCEvaluator, RKOCEvaluatorMFV

struct RKOCEvaluatorMFV{M,T,N}
    instructions::Vector{ButcherInstruction}
    output_indices::Vector{Int}
    inv_gamma::Vector{T}
    source_indices::Vector{Int}
    child_indices::Vector{Int}
    sibling_indices::Vector{Pair{Int,Int}}
    sibling_ranges::Vector{UnitRange{Int}}
    A_rows::Vector{MultiFloatVec{M,T,N}}
    A_cols::Vector{MultiFloatVec{M,T,N}}
    dA_cols::Vector{MultiFloatVec{M,T,N}}
    b::Array{MultiFloatVec{M,T,N},0}
    db::Array{MultiFloatVec{M,T,N},0}
    phi::Vector{MultiFloatVec{M,T,N}}
    dphi::Vector{MultiFloatVec{M,T,N}}
    residuals::Vector{MultiFloat{T,N}}
end

function RKOCEvaluatorMFV{M,T,N}(trees::Vector{LevelSequence}) where {M,T,N}
    instructions, output_indices = build_butcher_instruction_table(trees)
    inv_gamma = [inv(T(butcher_density(tree))) for tree in trees]
    source_indices = [-1 for _ in instructions]
    for (i, j) in enumerate(output_indices)
        source_indices[j] = i
    end
    child_indices, sibling_lists = compute_children_siblings(instructions)
    sibling_indices = reduce(vcat, sibling_lists)
    end_indices = cumsum(length.(sibling_lists))
    sibling_ranges = UnitRange{Int}.(
        vcat([0], end_indices[1:end-1]) .+ 1,
        end_indices
    )
    return RKOCEvaluatorMFV{M,T,N}(
        instructions,
        output_indices,
        inv_gamma,
        source_indices,
        child_indices,
        sibling_indices,
        sibling_ranges,
        Vector{MultiFloatVec{M,T,N}}(undef, M),
        Vector{MultiFloatVec{M,T,N}}(undef, M),
        Vector{MultiFloatVec{M,T,N}}(undef, M),
        Array{MultiFloatVec{M,T,N},0}(undef),
        Array{MultiFloatVec{M,T,N},0}(undef),
        Vector{MultiFloatVec{M,T,N}}(undef, length(instructions)),
        Vector{MultiFloatVec{M,T,N}}(undef, length(instructions)),
        Vector{MultiFloat{T,N}}(undef, length(output_indices)),
    )
end

RKOCEvaluatorMFV{M,T,N}(order::Int) where {M,T,N} =
    RKOCEvaluatorMFV{M,T,N}(all_rooted_trees(order))

###################################### EVALUATING BUTCHER WEIGHTS (FORWARD PASS)

function populate_phi!(evaluator::RKOCEvaluatorMFV{M,T,N}) where {M,T,N}
    instructions = evaluator.instructions
    A_cols = evaluator.A_cols
    phi = evaluator.phi
    @inbounds for (k, instruction) in enumerate(instructions)
        p = instruction.left
        q = instruction.right
        if p == -1
            phi[k] = one(MultiFloatVec{M,T,N})
        elseif q == -1
            phi_p = phi[p]
            phi[k] = +(ntuple(
                i -> A_cols[i] * MultiFloatVec{M,T,N}(phi_p[i]),
                Val{M}()
            )...)
        else
            phi[k] = phi[p] * phi[q]
        end
    end
    return nothing
end

function populate_residuals!(evaluator::RKOCEvaluatorMFV{M,T,N}) where {M,T,N}
    output_indices = evaluator.output_indices
    b = evaluator.b[]
    phi = evaluator.phi
    residuals = evaluator.residuals
    inv_gamma = evaluator.inv_gamma
    @inbounds for (i, k) in enumerate(output_indices)
        temp = b * phi[k]
        residuals[i] = +(ntuple(i -> temp[i], Val{M}())...) - inv_gamma[i]
    end
    return nothing
end

########################################### EVALUATING GRADIENTS (BACKWARD PASS)

function populate_dphi!(evaluator::RKOCEvaluatorMFV{M,T,N}) where {M,T,N}
    source_indices = evaluator.source_indices
    child_indices = evaluator.child_indices
    sibling_indices = evaluator.sibling_indices
    sibling_ranges = evaluator.sibling_ranges
    A_rows = evaluator.A_rows
    b = evaluator.b[]
    phi = evaluator.phi
    dphi = evaluator.dphi
    residuals = evaluator.residuals
    @inbounds begin
        for (k, s) in Iterators.reverse(enumerate(source_indices))
            if s == -1
                dphi[k] = zero(MultiFloatVec{M,T,N})
            else
                residual = residuals[s]
                temp = residual + residual
                dphi[k] = MultiFloatVec{M,T,N}(temp) * b
            end
            c = child_indices[k]
            if c != -1
                dphi_c = dphi[c]
                temp = dphi[k]
                for j = 1:M
                    temp += A_rows[j] * MultiFloatVec{M,T,N}(dphi_c[j])
                end
                dphi[k] = temp
            end
            for i in sibling_ranges[k]
                (p, q) = sibling_indices[i]
                dphi[k] += phi[p] * dphi[q]
            end
        end
        return nothing
    end
end

function populate_gradients!(evaluator::RKOCEvaluatorMFV{M,T,N}) where {M,T,N}
    output_indices = evaluator.output_indices
    child_indices = evaluator.child_indices
    dA_cols = evaluator.dA_cols
    phi = evaluator.phi
    dphi = evaluator.dphi
    residuals = evaluator.residuals
    for j = 1:M
        dA_cols[j] = zero(MultiFloatVec{M,T,N})
    end
    for (k, c) in enumerate(child_indices)
        if c != -1
            phi_k = phi[k]
            for t = 1:M
                dA_cols[t] += MultiFloatVec{M,T,N}(phi_k[t]) * dphi[c]
            end
        end
    end
    db = zero(MultiFloatVec{M,T,N})
    for (i, k) in enumerate(output_indices)
        db += MultiFloatVec{M,T,N}(residuals[i]) * phi[k]
    end
    evaluator.db[] = db + db
    return nothing
end

################################################################################

function (evaluator::RKOCEvaluatorMFV{M,T,N})(
    x::Vector{MultiFloat{T,N}}
) where {M,T,N}
    A_rows = evaluator.A_rows
    A_cols = evaluator.A_cols
    residuals = evaluator.residuals
    for i = 1:M
        A_rows[i] = MultiFloatVec{M,T,N}(ntuple(
            j -> (j < i) ? x[(((i-1)*(i-2))>>1)+j] : zero(MultiFloat{T,N}),
            Val{M}()
        ))
        A_cols[i] = MultiFloatVec{M,T,N}(ntuple(
            j -> (j > i) ? x[i+(((j-1)*(j-2))>>1)] : zero(MultiFloat{T,N}),
            Val{M}()
        ))
    end
    let i = M + 1
        evaluator.b[] = MultiFloatVec{M,T,N}(ntuple(
            j -> (j < i) ? x[(((i-1)*(i-2))>>1)+j] : zero(MultiFloat{T,N}),
            Val{M}()
        ))
    end
    populate_phi!(evaluator)
    populate_residuals!(evaluator)
    result = zero(MultiFloat{T,N})
    for i = 1:length(residuals)
        temp = @inbounds residuals[i]
        result += temp * temp
    end
    return result
end

struct RKOCEvaluatorMFVAdjoint{M,T,N}
    evaluator::RKOCEvaluatorMFV{M,T,N}
end

Base.adjoint(evaluator::RKOCEvaluatorMFV{M,T,N}) where {M,T,N} =
    RKOCEvaluatorMFVAdjoint{M,T,N}(evaluator)

function (adjoint::RKOCEvaluatorMFVAdjoint{M,T,N})(
    g::Vector{MultiFloat{T,N}},
    x::Vector{MultiFloat{T,N}}
) where {M,T,N}
    evaluator = adjoint.evaluator
    A_rows = evaluator.A_rows
    A_cols = evaluator.A_cols
    for i = 1:M
        A_rows[i] = MultiFloatVec{M,T,N}(ntuple(
            j -> (j < i) ? x[(((i-1)*(i-2))>>1)+j] : zero(MultiFloat{T,N}),
            Val{M}()
        ))
        A_cols[i] = MultiFloatVec{M,T,N}(ntuple(
            j -> (j > i) ? x[i+(((j-1)*(j-2))>>1)] : zero(MultiFloat{T,N}),
            Val{M}()
        ))
    end
    let i = M + 1
        evaluator.b[] = MultiFloatVec{M,T,N}(ntuple(
            j -> (j < i) ? x[(((i-1)*(i-2))>>1)+j] : zero(MultiFloat{T,N}),
            Val{M}()
        ))
    end
    populate_phi!(evaluator)
    populate_residuals!(evaluator)
    populate_dphi!(evaluator)
    populate_gradients!(evaluator)
    dA_cols = evaluator.dA_cols
    db = evaluator.db[]
    k = 0
    for i = 2:M
        @simd ivdep for j = 1:i-1
            @inbounds g[j+k] = dA_cols[j][i]
        end
        k += i - 1
    end
    @simd ivdep for i = 1:M
        @inbounds g[i+k] = db[i]
    end
    return g
end

function (adjoint::RKOCEvaluatorMFVAdjoint{M,T,N})(
    x::Vector{MultiFloat{T,N}}
) where {M,T,N}
    return adjoint(similar(x), x)
end

=#

end # module RungeKuttaToolKit
