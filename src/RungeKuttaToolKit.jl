module RungeKuttaToolKit


include("ButcherInstructions.jl")


#=

################################################## RKOC EVALUATOR DATA STRUCTURE


using MultiFloats


export RKOCEvaluator, RKOCEvaluatorMFV


struct RKOCEvaluator{T}
    num_stages::Int
    instructions::Vector{ButcherInstruction}
    output_indices::Vector{Int}
    inv_gamma::Vector{T}
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
    residuals::Vector{T}
end


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


function RKOCEvaluator{T}(
    trees::Vector{LevelSequence}, num_stages::Int
) where {T}
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
    return RKOCEvaluator{T}(
        num_stages,
        instructions,
        output_indices,
        inv_gamma,
        source_indices,
        child_indices,
        sibling_indices,
        sibling_ranges,
        Matrix{T}(undef, num_stages, num_stages),
        Matrix{T}(undef, num_stages, num_stages),
        Vector{T}(undef, num_stages),
        Vector{T}(undef, num_stages),
        Matrix{T}(undef, num_stages, length(instructions)),
        Matrix{T}(undef, num_stages, length(instructions)),
        Vector{T}(undef, length(output_indices)),
    )
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


RKOCEvaluator{T}(order::Int, num_stages::Int) where {T} =
    RKOCEvaluator{T}(all_rooted_trees(order), num_stages)


RKOCEvaluatorMFV{M,T,N}(order::Int) where {M,T,N} =
    RKOCEvaluatorMFV{M,T,N}(all_rooted_trees(order))


###################################### EVALUATING BUTCHER WEIGHTS (FORWARD PASS)


using LinearAlgebra: mul!


function populate_phi!(evaluator::RKOCEvaluator{T}) where {T}
    n = evaluator.num_stages
    instructions = evaluator.instructions
    A = evaluator.A
    phi = evaluator.phi
    @inbounds for (k, instruction) in enumerate(instructions)
        p = instruction.left
        q = instruction.right
        if p == -1
            @assert q == -1
            @simd ivdep for j = 1:n
                phi[j, k] = one(T)
            end
        elseif q == -1
            mul!(view(phi, :, k), A, view(phi, :, p))
        else
            @simd ivdep for j = 1:n
                phi[j, k] = phi[j, p] * phi[j, q]
            end
        end
    end
    return nothing
end


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


function populate_residuals!(evaluator::RKOCEvaluator{T}) where {T}
    n = evaluator.num_stages
    output_indices = evaluator.output_indices
    b = evaluator.b
    phi = evaluator.phi
    residuals = evaluator.residuals
    inv_gamma = evaluator.inv_gamma
    @inbounds for (i, k) in enumerate(output_indices)
        result = zero(T)
        @simd for j = 1:n
            result += b[j] * phi[j, k]
        end
        residuals[i] = result - inv_gamma[i]
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


################################################################################
################################################################################
################################################################################

module Legacy

export RKOCEvaluator, evaluate_residual!, evaluate_jacobian!,
    evaluate_error_coefficients!, evaluate_error_jacobian!,
    constrain!, compute_order!, compute_stages,
    RKOCBackpropEvaluator, evaluate_residual2, evaluate_gradient!,
    RKOCExplicitBackpropObjectiveFunctor,
    RKOCExplicitBackpropGradientFunctor,
    RKOCImplicitBackpropObjectiveFunctor,
    RKOCImplicitBackpropGradientFunctor,
    rkoc_explicit_backprop_functors,
    rkoc_implicit_backprop_functors

using Base.Threads: @threads, nthreads, threadid
using LinearAlgebra: mul!, ldiv!, qrfactUnblocked!

include("./ExampleMethods.jl")

##################################################################### PARTITIONS

function _int_parts_impl(n::Int, max::Int)::Vector{Vector{Pair{Int,Int}}}
    if n == 0
        [Pair{Int,Int}[]]
    elseif max == 0
        Vector{Pair{Int,Int}}[]
    else
        result = Vector{Pair{Int,Int}}[]
        for m = max:-1:1, q = div(n, m):-1:1
            for p in _int_parts_impl(n - q * m, m - 1)
                push!(result, push!(p, m => q))
            end
        end
        result
    end
end

function _int_parts_impl(n::Int, max::Int, len::Int)::Vector{Vector{Int}}
    if n == 0
        [zeros(Int, len)]
    elseif max * len < n
        Vector{Int}[]
    else
        result = Vector{Int}[]
        for m = min(n, max):-1:div(n + len - 1, len)
            for p in _int_parts_impl(n - m, m, len - 1)
                push!(result, push!(p, m))
            end
        end
        result
    end
end

integer_partitions(n::Int)::Vector{Vector{Pair{Int,Int}}} =
    reverse!.(_int_parts_impl(n, n))
integer_partitions(n::Int, len::Int)::Vector{Vector{Int}} =
    reverse!.(_int_parts_impl(n, n, len))

################################################################### COMBINATIONS

function previous_permutation!(items::Vector{T})::Bool where {T}
    num_items = length(items)
    if num_items == 0
        return false
    end
    current_item = items[num_items]
    pivot_index = num_items - 1
    while pivot_index != 0
        next_item = items[pivot_index]
        if next_item <= current_item
            pivot_index -= 1
            current_item = next_item
        else
            break
        end
    end
    if pivot_index == 0
        return false
    end
    pivot = items[pivot_index]
    successor_index = num_items
    while items[successor_index] >= pivot
        successor_index -= 1
    end
    items[pivot_index], items[successor_index] =
        items[successor_index], items[pivot_index]
    reverse!(view(items, pivot_index+1:num_items))
    return true
end

function combinations_with_replacement(
    items::Vector{T}, n::Int
)::Vector{Vector{Pair{T,Int}}} where {T}
    combinations = Vector{Pair{T,Int}}[]
    for p in integer_partitions(n, length(items))
        while true
            comb = Pair{T,Int}[]
            for (item, k) in zip(items, p)
                if k > 0
                    push!(comb, item => k)
                end
            end
            push!(combinations, comb)
            if !previous_permutation!(p)
                break
            end
        end
    end
    combinations
end

################################################################### ROOTED TREES

struct RootedTree
    order::Int
    children::Vector{Pair{RootedTree,Int}}
end

function rooted_trees(n::Int)::Vector{Vector{RootedTree}}
    result = Vector{RootedTree}[]
    for k = 1:n
        trees = RootedTree[]
        for partition in integer_partitions(k - 1)
            combination_candidates = [
                combinations_with_replacement(result[order], multiplicity)
                for (order, multiplicity) in partition]
            for combination_sequence in Base.product(combination_candidates...)
                push!(trees, RootedTree(k, vcat(combination_sequence...)))
            end
        end
        push!(result, trees)
    end
    result
end

function butcher_density(tree::RootedTree)::Int
    result = tree.order
    for (subtree, multiplicity) in tree.children
        result *= butcher_density(subtree)^multiplicity
    end
    result
end

function butcher_symmetry(tree::RootedTree)::Int
    result = 1
    for (subtree, multiplicity) in tree.children
        result *= butcher_symmetry(subtree) * factorial(multiplicity)
    end
    result
end

####################################################### ROOTED TREE MANIPULATION

function Base.show(io::IO, tree::RootedTree)::Nothing
    print(io, '[')
    for (subtree, multiplicity) in tree.children
        print(io, subtree)
        if multiplicity != 1
            print(io, '^', multiplicity)
        end
    end
    print(io, ']')
end

function tree_index_table(
    trees_of_order::Vector{Vector{RootedTree}})::Dict{String,Int}
    order = length(trees_of_order)
    tree_index = Dict{String,Int}()
    i = 0
    for p = 1:order-1, tree in trees_of_order[p]
        tree_index[string(tree)] = (i += 1)
    end
    tree_index
end

function dependency_table(
    trees_of_order::Vector{Vector{RootedTree}}
)::Vector{Vector{Int}}
    order = length(trees_of_order)
    tree_index = tree_index_table(trees_of_order)
    dependencies = Vector{Int}[]
    for p = 1:order, tree in trees_of_order[p]
        num_children = length(tree.children)
        if num_children == 0
            push!(dependencies, Int[])
        elseif num_children == 1
            child, m = tree.children[1]
            if m == 1
                a = tree_index[string(child)]
                push!(dependencies, [a])
            elseif m % 2 == 0
                half_child = RootedTree(0, [child => div(m, 2)])
                a = tree_index[string(half_child)]
                push!(dependencies, [a, a])
            else
                half_child = RootedTree(0, [child => div(m, 2)])
                half_plus_one_child = RootedTree(0, [child => div(m, 2) + 1])
                a = tree_index[string(half_child)]
                b = tree_index[string(half_plus_one_child)]
                push!(dependencies, [a, b])
            end
        else
            head_factor = RootedTree(0, tree.children[1:1])
            tail_factor = RootedTree(0, tree.children[2:end])
            a = tree_index[string(head_factor)]
            b = tree_index[string(tail_factor)]
            push!(dependencies, [a, b])
        end
    end
    dependencies
end

function generation_table(dependencies::Vector{Vector{Int}})::Vector{Int}
    generation = Vector{Int}(undef, length(dependencies))
    for (i, deps) in enumerate(dependencies)
        if length(deps) == 0
            generation[i] = 0
        else
            generation[i] = maximum(generation[j] for j in deps) + 1
        end
    end
    generation
end

function deficit_table(dependencies::Vector{Vector{Int}})::Vector{Int}
    deficit = Vector{Int}(undef, length(dependencies))
    for (i, deps) in enumerate(dependencies)
        if length(deps) == 0
            deficit[i] = 0
        elseif length(deps) == 1
            deficit[i] = deficit[deps[1]] + 1
        else
            deficit[i] = maximum(deficit[j] for j in deps)
        end
    end
    deficit
end

function range_table(
    dependencies::Vector{Vector{Int}},
    num_stages::Int
)::Vector{Tuple{Int,Int}}
    deficit = deficit_table(dependencies)
    lengths = num_stages .- deficit
    clamp!(lengths, 0, typemax(Int))
    lengths[1] = 0
    cumsum!(lengths, lengths)
    table_size = lengths[end]
    ranges = vcat([(1, 0)], collect(
        zip(lengths[1:end-1] .+ 1, lengths[2:end])))
end

function instruction_table(
    dependencies::Vector{Vector{Int}},
    ranges::Vector{Tuple{Int,Int}}
)::Vector{NTuple{5,Int}}
    instructions = Vector{NTuple{5,Int}}(undef, length(dependencies))
    for (i, deps) in enumerate(dependencies)
        num_deps = length(deps)
        a, b = ranges[i]
        if num_deps == 0
            instructions[i] = (i, a, b, -1, -1)
        elseif num_deps == 1
            c, d = ranges[deps[1]]
            instructions[i] = (i, a, b, c, -1)
        else
            c, d = ranges[deps[1]]
            e, f = ranges[deps[2]]
            instructions[i] = (i, a, b, d - (b - a), f - (b - a))
        end
    end
    instructions
end

############################################################# INDEX MANIPULATION

function require!(
    required::Vector{Bool},
    dependencies::Vector{Vector{Int}},
    index::Int
)::Nothing
    if !required[index]
        required[index] = true
        for dep in dependencies[index]
            require!(required, dependencies, dep)
        end
    end
end

function assert_requirements_satisfied(
    required::Vector{Bool},
    dependencies::Vector{Vector{Int}},
    indices::Vector{Int}
)::Nothing
    @assert(all(required[indices]))
    for index = 1:length(dependencies)
        if required[index]
            @assert(all(required[dep] for dep in dependencies[index]))
        end
    end
end

function included_indices(mask::Vector{Bool})::Vector{Int}
    result = Vector{Int}(undef, sum(mask))
    count = 0
    for index = 1:length(mask)
        if mask[index]
            result[count+=1] = index
        end
    end
    result
end

function restricted_indices(mask::Vector{Bool})::Vector{Int}
    result = Vector{Int}(undef, length(mask))
    count = 0
    for index = 1:length(mask)
        if mask[index]
            result[index] = (count += 1)
        else
            result[index] = 0
        end
    end
    result
end

function inverse_index_table(indices::Vector{Int}, n::Int)
    result = [0 for _ = 1:n]
    for (i, j) in enumerate(indices)
        result[j] = i
    end
    result
end

function restricted_trees_dependencies(indices::Vector{Int})
    max_index = maximum(indices)
    order = 1
    while max_index > rooted_tree_count(order)
        order += 1
    end
    trees_of_order = rooted_trees(order)
    full_dependencies = dependency_table(trees_of_order)
    required = [false for _ = 1:length(full_dependencies)]
    for index in indices
        require!(required, full_dependencies, index)
    end
    assert_requirements_satisfied(required, full_dependencies, indices)
    included = included_indices(required)
    restrict = restricted_indices(required)
    restricted_trees = vcat(trees_of_order...)[included]
    restricted_dependencies = Vector{Int}[]
    for index in included
        push!(restricted_dependencies, restrict[full_dependencies[index]])
    end
    (restricted_trees, restricted_dependencies,
        inverse_index_table(restrict[indices], length(included)))
end

##################################################### RKOC EVALUATOR CONSTRUCTOR

struct RKOCEvaluator{T<:Real}
    order::Int
    num_stages::Int
    num_vars::Int
    num_constrs::Int
    output_indices::Vector{Int}
    rounds::Vector{Vector{NTuple{5,Int}}}
    inv_density::Vector{T}
    inv_symmetry::Vector{T}
    u::Vector{T}
    vs::Vector{Vector{T}}
end

function RKOCEvaluator{T}(order::Int, num_stages::Int) where {T<:Real}
    # TODO: This assertion should not be necessary.
    @assert(order >= 2)
    num_vars = div(num_stages * (num_stages + 1), 2)
    trees_of_order = rooted_trees(order)
    num_constrs = sum(length.(trees_of_order))
    dependencies = dependency_table(trees_of_order)
    generation = generation_table(dependencies)
    ranges = range_table(dependencies, num_stages)
    table_size = ranges[end][2]
    instructions = instruction_table(dependencies, ranges)
    rounds = [NTuple{5,Int}[] for _ = 1:maximum(generation)-1]
    for (i, g) in enumerate(generation)
        if g > 1
            push!(rounds[g-1], instructions[i])
        end
    end
    return RKOCEvaluator{T}(
        order, num_stages, num_vars, num_constrs, Int[], rounds,
        inv.(T.(butcher_density.(vcat(trees_of_order...)))),
        inv.(T.(butcher_symmetry.(vcat(trees_of_order...)))),
        Vector{T}(undef, table_size),
        [Vector{T}(undef, table_size) for _ = 1:nthreads()]
    )
end

function RKOCEvaluator{T}(
    indices::Vector{Int}, num_stages::Int
) where {T<:Real}
    num_vars = div(num_stages * (num_stages + 1), 2)
    trees, dependencies, output_indices =
        restricted_trees_dependencies(indices)
    num_constrs = length(indices)
    generation = generation_table(dependencies)
    ranges = range_table(dependencies, num_stages)
    table_size = ranges[end][2]
    instructions = instruction_table(dependencies, ranges)
    rounds = [NTuple{5,Int}[] for _ = 1:maximum(generation)-1]
    for (i, g) in enumerate(generation)
        if g > 1
            push!(rounds[g-1], instructions[i])
        end
    end
    return RKOCEvaluator{T}(
        0, num_stages, num_vars, num_constrs,
        output_indices, rounds,
        inv.(T.(butcher_density.(trees))),
        inv.(T.(butcher_symmetry.(trees))),
        Vector{T}(undef, table_size),
        [Vector{T}(undef, table_size) for _ = 1:nthreads()]
    )
end

######################################################## RKOC EVALUATION HELPERS

function populate_u_init!(
    evaluator::RKOCEvaluator{T}, x::Vector{T}
)::Nothing where {T<:Real}
    u = evaluator.u
    k = 1
    for i = 1:evaluator.num_stages-1
        @inbounds result = x[k]
        @simd for j = 1:i-1
            @inbounds result += x[k+j]
        end
        @inbounds u[i] = result
        k += i
    end
end

function lvm_u!(
    evaluator::RKOCEvaluator{T}, dst_begin::Int, dst_end::Int,
    x::Vector{T}, src_begin::Int
)::Nothing where {T<:Real}
    if dst_begin > dst_end
        return
    end
    u = evaluator.u
    dst_size = dst_end - dst_begin + 1
    skip = evaluator.num_stages - dst_size
    index = div(skip * (skip + 1), 2)
    @inbounds u[dst_begin] = x[index] * u[src_begin]
    for i = 1:dst_size-1
        index += skip
        skip += 1
        @inbounds result = x[index] * u[src_begin]
        @simd for j = 1:i
            @inbounds result += x[index+j] * u[src_begin+j]
        end
        @inbounds u[dst_begin+i] = result
    end
end

function populate_u!(
    evaluator::RKOCEvaluator{T}, x::Vector{T}
)::Nothing where {T<:Real}
    u = evaluator.u
    populate_u_init!(evaluator, x)
    for round in evaluator.rounds
        @threads for (_, dst_begin, dst_end, src1, src2) in round
            if src2 == -1
                lvm_u!(evaluator, dst_begin, dst_end, x, src1)
            elseif src1 == src2
                @simd ivdep for i = 0:dst_end-dst_begin
                    @inbounds u[dst_begin+i] = u[src1+i]^2
                end
            else
                @simd ivdep for i = 0:dst_end-dst_begin
                    @inbounds u[dst_begin+i] = u[src1+i] * u[src2+i]
                end
            end
        end
    end
end

function populate_v_init!(
    evaluator::RKOCEvaluator{T}, v::Vector{T}, var_index::Int
)::Nothing where {T<:Real}
    j = 0
    for i = 1:evaluator.num_stages-1
        result = (j == var_index)
        j += 1
        for _ = 1:i-1
            result |= (j == var_index)
            j += 1
        end
        @inbounds v[i] = T(result)
    end
end

function lvm_v!(
    evaluator::RKOCEvaluator{T}, v::Vector{T}, var_index::Int,
    dst_begin::Int, dst_end::Int,
    x::Vector{T}, src_begin::Int
)::Nothing where {T<:Real}
    if dst_begin > dst_end
        return
    end
    u = evaluator.u
    dst_size = dst_end - dst_begin + 1
    skip = evaluator.num_stages - dst_size
    index = div(skip * (skip + 1), 2)
    if index == var_index + 1
        @inbounds v[dst_begin] = (x[index-1+1] * v[src_begin] +
                                  u[src_begin+var_index-index+1])
    else
        @inbounds v[dst_begin] = (x[index-1+1] * v[src_begin])
    end
    for i = 1:dst_size-1
        index += skip
        skip += 1
        @inbounds result = x[index] * v[src_begin]
        @simd for j = 1:i
            @inbounds result += x[index+j] * v[src_begin+j]
        end
        if (index <= var_index + 1) && (var_index + 1 <= index + i)
            @inbounds result += u[src_begin+var_index-index+1]
        end
        @inbounds v[dst_begin+i] = result
    end
end

function populate_v!(
    evaluator::RKOCEvaluator{T}, v::Vector{T},
    x::Vector{T}, var_index::Int
)::Nothing where {T<:Real}
    u = evaluator.u
    populate_v_init!(evaluator, v, var_index)
    for round in evaluator.rounds
        for (_, dst_begin, dst_end, src1, src2) in round
            if src2 == -1
                lvm_v!(evaluator, v, var_index, dst_begin, dst_end, x, src1)
            elseif src1 == src2
                @simd ivdep for i = 0:dst_end-dst_begin
                    temp = u[src1+i] * v[src1+i]
                    @inbounds v[dst_begin+i] = temp + temp
                end
            else
                @simd ivdep for i = 0:dst_end-dst_begin
                    @inbounds v[dst_begin+i] = (u[src1+i] * v[src2+i] +
                                                u[src2+i] * v[src1+i])
                end
            end
        end
    end
end

@inline function dot_inplace(
    n::Int,
    v::Vector{T}, v_offset::Int,
    w::Vector{T}, w_offset::Int
)::T where {T<:Real}
    result = zero(T)
    @simd for i = 1:n
        @inbounds result += v[v_offset+i] * w[w_offset+i]
    end
    result
end

################################################################ RKOC EVALUATION

function evaluate_residual!(
    res::Vector{T}, x::Vector{T},
    evaluator::RKOCEvaluator{T}
)::Nothing where {T<:Real}
    @assert(length(x) == evaluator.num_vars)
    u = evaluator.u
    num_stages = evaluator.num_stages
    num_vars = evaluator.num_vars
    output_indices = evaluator.output_indices
    inv_density = evaluator.inv_density
    populate_u!(evaluator, x)
    if length(output_indices) == 0
        let
            first = -one(T)
            b_idx = num_vars - num_stages + 1
            @simd for i = b_idx:num_vars
                @inbounds first += x[i]
            end
            @inbounds res[1] = first
        end
        @inbounds res[2] = dot_inplace(num_stages - 1, u, 0,
            x, num_vars - num_stages + 1) - T(0.5)
        for round in evaluator.rounds
            @threads for (res_index, dst_begin, dst_end, _, _) in round
                j = dst_begin - 1
                n = dst_end - j
                @inbounds res[res_index] =
                    dot_inplace(n, u, j, x, num_vars - n) -
                    inv_density[res_index]
            end
        end
    else
        let
            output_index = output_indices[1]
            if output_index > 0
                first = -one(T)
                b_idx = num_vars - num_stages + 1
                @simd for i = b_idx:num_vars
                    @inbounds first += x[i]
                end
                @inbounds res[output_index] = first
            end
        end
        let
            output_index = output_indices[2]
            if output_index > 0
                b_idx = num_vars - num_stages + 1
                @inbounds res[output_index] =
                    dot_inplace(num_stages - 1, u, 0, x, b_idx) - T(0.5)
            end
        end
        for round in evaluator.rounds
            @threads for (res_index, dst_begin, dst_end, _, _) in round
                output_index = output_indices[res_index]
                if output_index > 0
                    j = dst_begin - 1
                    n = dst_end - j
                    @inbounds res[output_index] =
                        dot_inplace(n, u, j, x, num_vars - n) -
                        inv_density[res_index]
                end
            end
        end
    end
end

function evaluate_error_coefficients!(
    res::Vector{T}, x::Vector{T},
    evaluator::RKOCEvaluator{T}
)::Nothing where {T<:Real}
    @assert(length(x) == evaluator.num_vars)
    u = evaluator.u
    num_stages = evaluator.num_stages
    num_vars = evaluator.num_vars
    output_indices = evaluator.output_indices
    inv_density = evaluator.inv_density
    inv_symmetry = evaluator.inv_symmetry
    populate_u!(evaluator, x)
    if length(output_indices) == 0
        let
            first = -one(T)
            b_idx = num_vars - num_stages + 1
            @simd for i = b_idx:num_vars
                @inbounds first += x[i]
            end
            @inbounds res[1] = first
        end
        @inbounds res[2] = dot_inplace(num_stages - 1, u, 0,
            x, num_vars - num_stages + 1) - T(0.5)
        for round in evaluator.rounds
            @threads for (res_index, dst_begin, dst_end, _, _) in round
                j = dst_begin - 1
                n = dst_end - j
                @inbounds res[res_index] = inv_symmetry[res_index] * (
                    dot_inplace(n, u, j, x, num_vars - n) -
                    inv_density[res_index])
            end
        end
    else
        let
            output_index = output_indices[1]
            if output_index > 0
                first = -one(T)
                b_idx = num_vars - num_stages + 1
                @simd for i = b_idx:num_vars
                    @inbounds first += x[i]
                end
                @inbounds res[output_index] = first
            end
        end
        let
            output_index = output_indices[2]
            if output_index > 0
                b_idx = num_vars - num_stages + 1
                @inbounds res[output_index] =
                    dot_inplace(num_stages - 1, u, 0, x, b_idx) - T(0.5)
            end
        end
        for round in evaluator.rounds
            @threads for (res_index, dst_begin, dst_end, _, _) in round
                output_index = output_indices[res_index]
                if output_index > 0
                    j = dst_begin - 1
                    n = dst_end - j
                    @inbounds res[output_index] = inv_symmetry[res_index] * (
                        dot_inplace(n, u, j, x, num_vars - n) -
                        inv_density[res_index])
                end
            end
        end
    end
end

########################################################### RKOC DIFFERENTIATION

function evaluate_jacobian!(
    jac::Matrix{T}, x::Vector{T},
    evaluator::RKOCEvaluator{T}
)::Nothing where {T<:Real}
    @assert(length(x) == evaluator.num_vars)
    u = evaluator.u
    num_stages = evaluator.num_stages
    num_vars = evaluator.num_vars
    output_indices = evaluator.output_indices
    populate_u!(evaluator, x)
    @threads for var_idx = 1:num_vars
        @inbounds v = evaluator.vs[threadid()]
        populate_v!(evaluator, v, x, var_idx - 1)
        if length(output_indices) == 0
            @inbounds jac[1, var_idx] = T(var_idx + num_stages > num_vars)
            let
                n = num_stages - 1
                m = num_vars - n
                result = dot_inplace(n, v, 0, x, m)
                if var_idx + n > num_vars
                    @inbounds result += u[var_idx-m]
                end
                @inbounds jac[2, var_idx] = result
            end
            for round in evaluator.rounds
                for (res_index, dst_begin, dst_end, _, _) in round
                    n = dst_end - dst_begin + 1
                    m = num_vars - n
                    result = dot_inplace(n, v, dst_begin - 1, x, m)
                    if var_idx + n > num_vars
                        @inbounds result += u[dst_begin-1+var_idx-m]
                    end
                    @inbounds jac[res_index, var_idx] = result
                end
            end
        else
            let
                output_index = output_indices[1]
                if output_index > 0
                    @inbounds jac[output_index, var_idx] =
                        T(var_idx + num_stages > num_vars)
                end
            end
            let
                output_index = output_indices[2]
                if output_index > 0
                    n = num_stages - 1
                    m = num_vars - n
                    result = dot_inplace(n, v, 0, x, m)
                    if var_idx + n > num_vars
                        @inbounds result += u[var_idx-m]
                    end
                    @inbounds jac[output_index, var_idx] = result
                end
            end
            for round in evaluator.rounds
                for (res_index, dst_begin, dst_end, _, _) in round
                    output_index = output_indices[res_index]
                    if output_index > 0
                        n = dst_end - dst_begin + 1
                        m = num_vars - n
                        result = dot_inplace(n, v, dst_begin - 1, x, m)
                        if var_idx + n > num_vars
                            @inbounds result += u[dst_begin-1+var_idx-m]
                        end
                        @inbounds jac[output_index, var_idx] = result
                    end
                end
            end
        end
    end
end

function evaluate_error_jacobian!(
    jac::Matrix{T}, x::Vector{T},
    evaluator::RKOCEvaluator{T}
)::Nothing where {T<:Real}
    @assert(length(x) == evaluator.num_vars)
    u = evaluator.u
    num_stages = evaluator.num_stages
    num_vars = evaluator.num_vars
    output_indices = evaluator.output_indices
    inv_symmetry = evaluator.inv_symmetry
    populate_u!(evaluator, x)
    @threads for var_idx = 1:num_vars
        @inbounds v = evaluator.vs[threadid()]
        populate_v!(evaluator, v, x, var_idx - 1)
        if length(output_indices) == 0
            @inbounds jac[1, var_idx] = T(var_idx + num_stages > num_vars)
            let
                n = num_stages - 1
                m = num_vars - n
                result = dot_inplace(n, v, 0, x, m)
                if var_idx + n > num_vars
                    @inbounds result += u[var_idx-m]
                end
                @inbounds jac[2, var_idx] = result
            end
            for round in evaluator.rounds
                for (res_index, dst_begin, dst_end, _, _) in round
                    n = dst_end - dst_begin + 1
                    m = num_vars - n
                    result = dot_inplace(n, v, dst_begin - 1, x, m)
                    if var_idx + n > num_vars
                        @inbounds result += u[dst_begin-1+var_idx-m]
                    end
                    result *= inv_symmetry[res_index]
                    @inbounds jac[res_index, var_idx] = result
                end
            end
        else
            let
                output_index = output_indices[1]
                if output_index > 0
                    @inbounds jac[output_index, var_idx] =
                        T(var_idx + num_stages > num_vars)
                end
            end
            let
                output_index = output_indices[2]
                if output_index > 0
                    n = num_stages - 1
                    m = num_vars - n
                    result = dot_inplace(n, v, 0, x, m)
                    if var_idx + n > num_vars
                        @inbounds result += u[var_idx-m]
                    end
                    @inbounds jac[output_index, var_idx] = result
                end
            end
            for round in evaluator.rounds
                for (res_index, dst_begin, dst_end, _, _) in round
                    output_index = output_indices[res_index]
                    if output_index > 0
                        n = dst_end - dst_begin + 1
                        m = num_vars - n
                        result = dot_inplace(n, v, dst_begin - 1, x, m)
                        if var_idx + n > num_vars
                            @inbounds result += u[dst_begin-1+var_idx-m]
                        end
                        result *= inv_symmetry[res_index]
                        @inbounds jac[output_index, var_idx] = result
                    end
                end
            end
        end
    end
end

############################################################## FUNCTOR INTERFACE

struct RKOCEvaluatorAdjointProxy{T<:Real}
    evaluator::RKOCEvaluator{T}
end

function Base.adjoint(evaluator::RKOCEvaluator{T}) where {T<:Real}
    RKOCEvaluatorAdjointProxy{T}(evaluator)
end

function (evaluator::RKOCEvaluator{T})(x::Vector{T}) where {T<:Real}
    residual = Vector{T}(undef, evaluator.num_constrs)
    evaluate_error_coefficients!(residual, x, evaluator)
    residual
end

function (proxy::RKOCEvaluatorAdjointProxy{T})(x::Vector{T}) where {T<:Real}
    jacobian = Matrix{T}(undef,
        proxy.evaluator.num_constrs, proxy.evaluator.num_vars)
    evaluate_error_jacobian!(jacobian, x, proxy.evaluator)
    jacobian
end

############################################################## METHOD INSPECTION

@inline function norm2(x::Vector{T}, n::Int)::T where {T<:Number}
    result = zero(float(real(T)))
    @simd for i = 1:n
        @inbounds result += abs2(x[i])
    end
    result
end

function constrain!(
    x::Vector{T}, evaluator::RKOCEvaluator{T}
)::T where {T<:Real}
    num_vars, num_constrs = evaluator.num_vars, evaluator.num_constrs
    x_new = Vector{T}(undef, num_vars)
    residual = Vector{T}(undef, num_constrs)
    jacobian = Matrix{T}(undef, num_constrs, num_vars)
    direction = Vector{T}(undef, num_vars)
    evaluate_error_coefficients!(residual, x, evaluator)
    obj_old = norm2(residual, num_constrs)
    while true
        evaluate_error_jacobian!(jacobian, x, evaluator)
        ldiv!(direction, qrfactUnblocked!(jacobian), residual)
        @simd ivdep for i = 1:num_vars
            @inbounds x_new[i] = x[i] - direction[i]
        end
        evaluate_error_coefficients!(residual, x_new, evaluator)
        obj_new = norm2(residual, num_constrs)
        if obj_new < obj_old
            @simd ivdep for i = 1:num_vars
                @inbounds x[i] = x_new[i]
            end
            obj_old = obj_new
        else
            return sqrt(obj_old)
        end
    end
end

function compute_order!(
    x::Vector{T}, threshold::T
)::Int where {T<:Real}
    num_stages = compute_stages(x)
    order = 2
    while true
        evaluator = RKOCEvaluator{T}(order, num_stages)
        obj_new = constrain!(x, evaluator)
        if obj_new <= threshold
            order += 1
        else
            return order - 1
        end
    end
end

function compute_stages(x::Vector{T})::Int where {T<:Real}
    num_stages = div(isqrt(8 * length(x) + 1) - 1, 2)
    @assert(length(x) == div(num_stages * (num_stages + 1), 2))
    num_stages
end

end # module Legacy

=#

end # module RungeKuttaToolKit
