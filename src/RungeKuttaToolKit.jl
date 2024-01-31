module RungeKuttaToolKit


include("ButcherInstructions.jl")
using .ButcherInstructions: LevelSequence, ButcherInstruction,
    all_rooted_trees, butcher_density, butcher_symmetry,
    build_instruction_table, compute_children_siblings


################################################################################


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


################################################################################


using LinearAlgebra: mul!


function populate_phi!(ev::RKOCEvaluator{T}) where {T}
    s = size(ev.phi, 1)
    @assert (s, s) == size(ev.A)
    @inbounds for (k, instruction) in enumerate(ev.instructions)
        p, q = instruction.left, instruction.right
        if p == -1
            @assert q == -1
            _one = one(T)
            @simd ivdep for j = 1:s
                ev.phi[j, k] = _one
            end
        elseif q == -1
            mul!(view(ev.phi, :, k), ev.A, view(ev.phi, :, p))
        else
            @simd ivdep for j = 1:s
                ev.phi[j, k] = ev.phi[j, p] * ev.phi[j, q]
            end
        end
    end
    return nothing
end


function populate_residuals!(ev::RKOCEvaluator{T}) where {T}
    s = size(ev.phi, 1)
    @assert (s,) == size(ev.b)
    @inbounds for (i, k) in enumerate(ev.output_indices)
        result = zero(T)
        @simd for j = 1:s
            result += ev.b[j] * ev.phi[j, k]
        end
        ev.residuals[i] = result - ev.inv_gamma[i]
    end
    return nothing
end


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


function populate_dphi!(evaluator::RKOCEvaluator{T}) where {T}
    n = evaluator.num_stages
    source_indices = evaluator.source_indices
    child_indices = evaluator.child_indices
    sibling_indices = evaluator.sibling_indices
    sibling_ranges = evaluator.sibling_ranges
    A = evaluator.A
    b = evaluator.b
    phi = evaluator.phi
    dphi = evaluator.dphi
    residuals = evaluator.residuals
    @inbounds begin
        for (k, s) in Iterators.reverse(enumerate(source_indices))
            if s == -1
                @simd ivdep for j = 1:n
                    dphi[j, k] = zero(T)
                end
            else
                residual = residuals[s]
                temp = residual + residual
                @simd ivdep for j = 1:n
                    dphi[j, k] = temp * b[j]
                end
            end
            c = child_indices[k]
            if c != -1
                for j = 1:n
                    temp = dphi[j, k]
                    @simd for i = 1:n
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
        return nothing
    end
end


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


function populate_gradients!(evaluator::RKOCEvaluator{T}) where {T}
    n = evaluator.num_stages
    output_indices = evaluator.output_indices
    child_indices = evaluator.child_indices
    dA = evaluator.dA
    db = evaluator.db
    phi = evaluator.phi
    dphi = evaluator.dphi
    residuals = evaluator.residuals
    for j = 1:n
        @simd ivdep for i = 1:n
            dA[i, j] = zero(T)
        end
    end
    for (k, c) in enumerate(child_indices)
        if c != -1
            for t = 1:n
                temp = phi[t, k]
                @simd ivdep for s = 1:n
                    dA[s, t] += temp * dphi[s, c]
                end
            end
        end
    end
    @simd ivdep for i = 1:n
        db[i] = zero(T)
    end
    for (i, k) in enumerate(output_indices)
        temp = residuals[i]
        @simd ivdep for j = 1:n
            db[j] += temp * phi[j, k]
        end
    end
    @simd ivdep for i = 1:n
        db[i] += db[i]
    end
    return nothing
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


################################################### RESHAPING COEFFICIENT ARRAYS


function populate_explicit!(A::Matrix{T}, b::Vector{T}, x::Vector{T}) where {T}
    n = length(b)
    @assert size(A) == (n, n)
    @assert length(x) == ((n * (n + 1)) >> 1)
    k = 0
    for i = 1:n
        @simd ivdep for j = 1:i-1
            @inbounds A[i, j] = x[j+k]
        end
        k += i - 1
        @simd ivdep for j = i:n
            @inbounds A[i, j] = zero(T)
        end
    end
    @simd ivdep for i = 1:n
        @inbounds b[i] = x[i+k]
    end
    return nothing
end


function populate_explicit!(x::Vector{T}, A::Matrix{T}, b::Vector{T}) where {T}
    n = length(b)
    @assert size(A) == (n, n)
    @assert length(x) == ((n * (n + 1)) >> 1)
    k = 0
    for i = 2:n
        @simd ivdep for j = 1:i-1
            @inbounds x[j+k] = A[i, j]
        end
        k += i - 1
    end
    @simd ivdep for i = 1:n
        @inbounds x[i+k] = b[i]
    end
    return nothing
end


function populate_implicit!(A::Matrix{T}, b::Vector{T}, x::Vector{T}) where {T}
    n = length(b)
    @assert size(A) == (n, n)
    @assert length(x) == n * (n + 1)
    n2 = n * n
    @simd ivdep for i = 1:n2
        @inbounds A[i] = x[i]
    end
    @simd ivdep for i = 1:n
        @inbounds b[i] = x[n2+i]
    end
    return nothing
end


function populate_implicit!(x::Vector{T}, A::Matrix{T}, b::Vector{T}) where {T}
    n = length(b)
    @assert size(A) == (n, n)
    @assert length(x) == n * (n + 1)
    n2 = n * n
    @simd ivdep for i = 1:n2
        @inbounds x[i] = A[i]
    end
    @simd ivdep for i = 1:n
        @inbounds x[n2+i] = b[i]
    end
    return nothing
end


################################################################################


function (evaluator::RKOCEvaluator{T})(x::Vector{T}) where {T}
    A = evaluator.A
    b = evaluator.b
    residuals = evaluator.residuals
    populate_explicit!(A, b, x)
    populate_phi!(evaluator)
    populate_residuals!(evaluator)
    result = zero(T)
    @simd for i = 1:length(residuals)
        temp = @inbounds residuals[i]
        result += temp * temp
    end
    return result
end


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
    @simd for i = 1:length(residuals)
        temp = @inbounds residuals[i]
        result += temp * temp
    end
    return result
end


struct RKOCEvaluatorAdjoint{T}
    evaluator::RKOCEvaluator{T}
end


struct RKOCEvaluatorMFVAdjoint{M,T,N}
    evaluator::RKOCEvaluatorMFV{M,T,N}
end


Base.adjoint(evaluator::RKOCEvaluator{T}) where {T} =
    RKOCEvaluatorAdjoint{T}(evaluator)


Base.adjoint(evaluator::RKOCEvaluatorMFV{M,T,N}) where {M,T,N} =
    RKOCEvaluatorMFVAdjoint{M,T,N}(evaluator)


function (adjoint::RKOCEvaluatorAdjoint{T})(g::Vector{T}, x::Vector{T}) where {T}
    evaluator = adjoint.evaluator
    A = evaluator.A
    dA = evaluator.dA
    b = evaluator.b
    db = evaluator.db
    populate_explicit!(A, b, x)
    populate_phi!(evaluator)
    populate_residuals!(evaluator)
    populate_dphi!(evaluator)
    populate_gradients!(evaluator)
    populate_explicit!(g, dA, db)
    return g
end


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


function (adjoint::RKOCEvaluatorAdjoint{T})(x::Vector{T}) where {T}
    return adjoint(similar(x), x)
end


function (adjoint::RKOCEvaluatorMFVAdjoint{M,T,N})(
    x::Vector{MultiFloat{T,N}}
) where {M,T,N}
    return adjoint(similar(x), x)
end

=#

end # module RungeKuttaToolKit
