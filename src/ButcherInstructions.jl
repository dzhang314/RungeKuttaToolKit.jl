module ButcherInstructions


################################################################### BIPARTITIONS


struct Bipartition{T}
    left::Vector{T}
    right::Vector{T}
end


struct BipartitionIterator{T}
    items::Vector{T}
end


Base.eltype(::Type{BipartitionIterator{T}}) where {T} = Bipartition{T}


function Base.length(iter::BipartitionIterator{T}) where {T}
    n = length(iter.items)
    if n <= 1
        return 1
    else
        return (1 << (length(iter.items) - 1)) - 1
    end
end


function Base.iterate(iter::BipartitionIterator{T}) where {T}
    n = length(iter.items)
    if iszero(n)
        return (Bipartition(T[], T[]), zero(UInt))
    end
    index = (one(UInt) << (n - 1)) - one(UInt)
    left = T[@inbounds iter.items[1]]
    right = Vector{T}(undef, n - 1)
    @simd ivdep for i = 1:n-1
        @inbounds right[i] = iter.items[i+1]
    end
    return (Bipartition(left, right), index)
end


function Base.iterate(iter::BipartitionIterator{T}, index::UInt) where {T}
    if iszero(index & ~one(UInt))
        return nothing
    end
    n = length(iter.items)
    index -= one(UInt)
    next_index = index
    left = T[@inbounds iter.items[1]]
    right = T[]
    @inbounds for i = 2:n
        push!(ifelse(iszero(index & one(UInt)), left, right), iter.items[i])
        index >>= one(UInt)
    end
    return (Bipartition(left, right), next_index)
end


################################################################ LEVEL SEQUENCES


struct LevelSequence
    data::Vector{Int}
end


Base.length(s::LevelSequence) = length(s.data)
Base.getindex(s::LevelSequence, i::Int) = s.data[i]
Base.:(==)(s::LevelSequence, t::LevelSequence) = (s.data == t.data)


function count_legs(s::LevelSequence)
    n = length(s)
    @assert n > 0
    @inbounds begin
        @assert isone(s[1])
        result = 0
        for i = 2:n
            result += (s[i] == 2)
        end
        return result
    end
end


function extract_legs(s::LevelSequence)
    n = length(s)
    @assert n > 0
    @inbounds begin
        @assert s[1] == 1
        result = Vector{LevelSequence}(undef, count_legs(s))
        if n > 1
            i = 1
            last = 2
            for j = 3:n
                if s[j] == 2
                    result[i] = LevelSequence(s.data[last:j-1])
                    result[i].data .-= 1
                    i += 1
                    last = j
                end
            end
            result[i] = LevelSequence(s.data[last:end])
            result[i].data .-= 1
        end
        return result
    end
end


function is_canonical(s::LevelSequence)
    legs = extract_legs(s)
    if !issorted(legs; by=(t -> t.data), rev=true)
        return false
    end
    for leg in legs
        if !is_canonical(leg)
            return false
        end
    end
    return true
end


#################################################### LEVEL SEQUENCE DICTIONARIES


struct LevelSequenceDict
    entries::Dict{Vector{Int},Int}
end


LevelSequenceDict() = LevelSequenceDict(Dict{Vector{Int},Int}())


Base.haskey(d::LevelSequenceDict, s::LevelSequence) =
    haskey(d.entries, s.data)
Base.getindex(d::LevelSequenceDict, s::LevelSequence) =
    getindex(d.entries, s.data)
Base.setindex!(d::LevelSequenceDict, i::Int, s::LevelSequence) =
    setindex!(d.entries, i, s.data)


##################################################### GENERATING LEVEL SEQUENCES


struct LevelSequenceIterator
    n::Int
end


Base.eltype(::Type{LevelSequenceIterator}) = LevelSequence


function Base.length(iter::LevelSequenceIterator)
    n = iter.n
    if n <= 0
        return 0
    else
        x = Vector{Int}(undef, n)
        @inbounds begin
            x[1] = 1
            for i = 2:n
                result = 0
                for j = 1:i-1
                    accumulator = 0
                    for d = 1:j
                        if j % d == 0
                            accumulator += d * x[d]
                        end
                    end
                    result += accumulator * x[i-j]
                end
                x[i] = div(result, i - 1)
            end
            return x[n]
        end
    end
end


function Base.iterate(iter::LevelSequenceIterator)
    # This function is based on the GENERATE-FIRST-TREE
    # algorithm from Figure 3 of the following paper:
    # CONSTANT TIME GENERATION OF ROOTED TREES
    # TERRY BEYER AND SANDRA MITCHELL HEDETNIEMI
    # SIAM J. COMPUT. Vol. 9, No. 4, November 1980
    n = iter.n
    if n <= 0
        return nothing
    end
    L = Vector{Int}(undef, n)
    PREV = Vector{Int}(undef, n - 1)
    SAVE = Vector{Int}(undef, n - 1)
    @inbounds begin
        @simd ivdep for i = 1:n
            L[i] = i
        end
        @simd ivdep for i = 1:n-1
            PREV[i] = i
        end
        @simd ivdep for i = 1:n-1
            SAVE[i] = 0
        end
    end
    return (LevelSequence(copy(L)), (L, PREV, SAVE, ifelse(n <= 2, 1, n)))
end


function Base.iterate(
    iter::LevelSequenceIterator,
    state::Tuple{Vector{Int},Vector{Int},Vector{Int},Int}
)
    # This function is based on the GENERATE-NEXT-TREE
    # algorithm from Figure 3 of the following paper:
    # CONSTANT TIME GENERATION OF ROOTED TREES
    # TERRY BEYER AND SANDRA MITCHELL HEDETNIEMI
    # SIAM J. COMPUT. Vol. 9, No. 4, November 1980
    (L, PREV, SAVE, p) = state
    if p == 1
        return nothing
    end
    n = iter.n
    @inbounds begin
        L[p] -= 1
        if (p < n) && (L[p] != 2 || L[p-1] != 2)
            diff = p - PREV[L[p]]
            while p < n
                SAVE[p] = PREV[L[p]]
                PREV[L[p]] = p
                p += 1
                L[p] = L[p-diff]
            end
        end
        while L[p] == 2
            p -= 1
            PREV[L[p]] = SAVE[p]
        end
    end
    return (LevelSequence(copy(L)), (L, PREV, SAVE, p))
end


function rooted_trees(n::Int; tree_ordering::Symbol=:reverse_lexicographic)
    if tree_ordering == :lexicographic
        return reverse!(collect(LevelSequenceIterator(n)))
    else
        return collect(LevelSequenceIterator(n))
    end
end


all_rooted_trees(n::Int; tree_ordering::Symbol=:reverse_lexicographic) =
    reduce(vcat, rooted_trees(i; tree_ordering=tree_ordering) for i = 1:n)


################################################## COMBINATORICS OF ROOTED TREES


function butcher_density(tree::LevelSequence)
    result = length(tree)
    for leg in extract_legs(tree)
        result *= butcher_density(leg)
    end
    return result
end


function butcher_symmetry(tree::LevelSequence)
    leg_counts = LevelSequenceDict()
    for leg in extract_legs(tree)
        if haskey(leg_counts, leg)
            leg_counts[leg] += 1
        else
            leg_counts[leg] = 1
        end
    end
    result = 1
    for (data, count) in leg_counts.entries
        result *= butcher_symmetry(LevelSequence(data)) * factorial(count)
    end
    return result
end


########################################################### BUTCHER INSTRUCTIONS


struct ButcherInstruction
    left::Int
    right::Int
    depth::Int
    deficit::Int
end


depth(instruction::ButcherInstruction) = instruction.depth


function find_butcher_instruction(
    instructions::Vector{ButcherInstruction},
    indices::LevelSequenceDict, tree::LevelSequence
)
    legs = extract_legs(tree)
    if isempty(legs)
        return ButcherInstruction(-1, -1, 1, 0)
    elseif isone(length(legs))
        leg = only(legs)
        if haskey(indices, leg)
            index = indices[leg]
            instruction = instructions[index]
            return ButcherInstruction(
                index, -1, instruction.depth + 1, instruction.deficit + 1)
        else
            return nothing
        end
    else
        candidates = ButcherInstruction[]
        for p in BipartitionIterator(legs)
            left_leg = LevelSequence(reduce(vcat,
                (leg.data .+ 1 for leg in p.left); init=[1]))
            if haskey(indices, left_leg)
                right_leg = LevelSequence(reduce(vcat,
                    (leg.data .+ 1 for leg in p.right); init=[1]))
                if haskey(indices, right_leg)
                    left_index = indices[left_leg]
                    right_index = indices[right_leg]
                    left_instruction = instructions[left_index]
                    right_instruction = instructions[right_index]
                    depth = 1 + max(
                        left_instruction.depth, right_instruction.depth)
                    deficit = max(
                        left_instruction.deficit, right_instruction.deficit)
                    push!(candidates, ButcherInstruction(
                        left_index, right_index, depth, deficit))
                end
            end
        end
        if isempty(candidates)
            return nothing
        else
            return first(sort!(candidates; by=depth))
        end
    end
end


function push_butcher_instructions!(
    instructions::Vector{ButcherInstruction},
    indices::LevelSequenceDict, tree::LevelSequence
)
    @assert !haskey(indices, tree)
    instruction = find_butcher_instruction(instructions, indices, tree)
    if isnothing(instruction)
        legs = extract_legs(tree)
        @assert !isempty(legs)
        if isone(length(legs))
            push_butcher_instructions!(instructions, indices, only(legs))
        else
            # TODO: What is the optimal policy for splitting a tree into left
            # and right subtrees? Ideally, we would like subtrees to be reused,
            # and we also want to minimize depth for parallelism. For now, we
            # always put one leg on the left and remaining legs on the right.
            left_leg = LevelSequence(vcat([1], legs[1].data .+ 1))
            right_leg = LevelSequence(reduce(vcat,
                (leg.data .+ 1 for leg in legs[2:end]); init=[1]))
            if !haskey(indices, left_leg)
                push_butcher_instructions!(instructions, indices, left_leg)
            end
            if !haskey(indices, right_leg)
                push_butcher_instructions!(instructions, indices, right_leg)
            end
        end
    end
    instruction = find_butcher_instruction(instructions, indices, tree)
    @assert !isnothing(instruction)
    push!(instructions, instruction)
    indices[tree] = length(instructions)
    return instruction
end


permute_butcher_instruction(
    instruction::ButcherInstruction, permutation::Vector{Int}
) = ButcherInstruction(
    (instruction.left == -1) ? -1 : permutation[instruction.left],
    (instruction.right == -1) ? -1 : permutation[instruction.right],
    instruction.depth, instruction.deficit)


function push_necessary_subtrees!(
    result::LevelSequenceDict, tree::LevelSequence
)
    result[tree] = 0
    legs = extract_legs(tree)
    if isone(length(legs))
        push_necessary_subtrees!(result, only(legs))
    else
        for leg in legs
            push_necessary_subtrees!(result,
                LevelSequence(vcat([1], leg.data .+ 1)))
        end
    end
    return result
end


function grevlex_order(a::LevelSequence, b::LevelSequence)
    len_a = length(a)
    len_b = length(b)
    if len_a < len_b
        return true
    elseif len_a > len_b
        return false
    else
        return a.data > b.data
    end
end


function necessary_subtrees(trees::Vector{LevelSequence})
    result = LevelSequenceDict()
    for tree in trees
        result[tree] = 0
        push_necessary_subtrees!(result, tree)
    end
    return sort!(LevelSequence.(keys(result.entries)); lt=grevlex_order)
end


function build_instructions(
    trees::Vector{LevelSequence};
    optimize::Bool=true, sort_by_depth::Bool=true
)
    @assert allunique(trees)
    instructions = ButcherInstruction[]
    indices = LevelSequenceDict()
    for tree in (optimize ? necessary_subtrees(trees) : trees)
        push_butcher_instructions!(instructions, indices, tree)
    end
    if sort_by_depth
        permutation = sortperm(instructions; by=depth)
        inverse_permutation = invperm(permutation)
        return ([permute_butcher_instruction(instruction, inverse_permutation)
                 for instruction in instructions[permutation]],
            [inverse_permutation[indices[tree]] for tree in trees])
    else
        return (instructions, [indices[tree] for tree in trees])
    end
end


function execute_instructions(instructions::Vector{ButcherInstruction})
    result = LevelSequence[]
    for instruction in instructions
        if instruction.left == -1
            @assert instruction.right == -1
            push!(result, LevelSequence([1]))
        elseif instruction.right == -1
            push!(result, LevelSequence(vcat([1],
                result[instruction.left].data .+ 1)))
        else
            legs = vcat(
                extract_legs(result[instruction.left]),
                extract_legs(result[instruction.right]))
            children = [leg.data .+ 1 for leg in legs]
            sort!(children; rev=true)
            push!(result, LevelSequence(reduce(vcat, children; init=[1])))
        end
    end
    @assert all(is_canonical, result)
    return result
end


##################################################### BUTCHER INSTRUCTION TABLES


struct ButcherInstructionTable
    instructions::Vector{ButcherInstruction}
    output_indices::Vector{Int}
    source_indices::Vector{Int}
    child_indices::Vector{Int}
    sibling_indices::Vector{Pair{Int,Int}}
    sibling_ranges::Vector{UnitRange{Int}}
end


function compute_children_siblings(instructions::Vector{ButcherInstruction})
    children = [-1 for _ in instructions]
    siblings = [Pair{Int,Int}[] for _ in instructions]
    for (i, instruction) in enumerate(instructions)
        if instruction.left == -1
            @assert instruction.right == -1
        elseif instruction.right == -1
            children[instruction.left] = i
        else
            push!(siblings[instruction.left], instruction.right => i)
            push!(siblings[instruction.right], instruction.left => i)
        end
    end
    return (children, siblings)
end


function ButcherInstructionTable(trees::Vector{LevelSequence})
    instructions, output_indices = build_instructions(trees)
    source_indices = [-1 for _ in instructions]
    for (i, j) in enumerate(output_indices)
        source_indices[j] = i
    end
    child_indices, sibling_lists = compute_children_siblings(instructions)
    sibling_indices = reduce(vcat, sibling_lists)
    end_indices = cumsum(length.(sibling_lists))
    start_indices = vcat([1], end_indices[1:end-1] .+ 1)
    sibling_ranges = UnitRange{Int}.(start_indices, end_indices)
    return ButcherInstructionTable(
        instructions, output_indices, source_indices,
        child_indices, sibling_indices, sibling_ranges)
end


end # module ButcherInstructions
