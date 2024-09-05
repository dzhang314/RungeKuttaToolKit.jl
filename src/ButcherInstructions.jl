module ButcherInstructions


using Printf
using ..RungeKuttaToolKit: NULL_INDEX


###################################################################### UTILITIES


function counts(items::AbstractVector{T}) where {T}
    result = Dict{T,Int}()
    for item in items
        if haskey(result, item)
            result[item] += 1
        else
            result[item] = 1
        end
    end
    return result
end


function add_dict!(dict::Dict{K,Set{V}}, key::K, value::V) where {K,V}
    if !haskey(dict, key)
        dict[key] = Set{V}()
    end
    push!(dict[key], value)
    return dict
end


################################################################### BIPARTITIONS


struct BipartitionIterator{T}
    items::Vector{T}
end


@inline Base.eltype(::Type{BipartitionIterator{T}}) where {T} =
    Tuple{Vector{T},Vector{T}}


@inline Base.length(iter::BipartitionIterator{T}) where {T} =
    isempty(iter.items) ? 0 : (1 << (length(iter.items) - 1)) - 1


function Base.iterate(iter::BipartitionIterator{T}) where {T}
    @inbounds begin
        n = length(iter.items)
        return n < 2 ? nothing : (
            ((T[iter.items[1]], iter.items[2:n])),
            (one(UInt) << (n - 1)) - 1)
    end
end


function Base.iterate(iter::BipartitionIterator{T}, state::UInt) where {T}
    @inbounds begin
        if state < 2
            return nothing
        end
        state -= 1
        next_state = state
        left = T[iter.items[1]]
        right = T[]
        for i = 2:length(iter.items)
            push!(ifelse(iszero(state & 1), left, right), iter.items[i])
            state >>= 1
        end
        return ((left, right), next_state)
    end
end


################################################################ LEVEL SEQUENCES


struct LevelSequence
    data::Vector{Int}
end


@inline Base.length(tree::LevelSequence) = length(tree.data)
@inline Base.getindex(tree::LevelSequence, i::Int) = tree.data[i]
@inline Base.copy(tree::LevelSequence) = LevelSequence(copy(tree.data))
@inline Base.hash(tree::LevelSequence, h::UInt) = hash(tree.data, h)
@inline Base.:(==)(s::LevelSequence, t::LevelSequence) = (s.data == t.data)
@inline Base.isless(s::LevelSequence, t::LevelSequence) = (s.data < t.data)


###################################################### SPLITTING LEVEL SEQUENCES


@inline count_legs(tree::LevelSequence) = count(==(2), tree.data)


@inline function extract_leg(tree::LevelSequence, i::Int, j::Int)
    @inbounds begin
        result = Vector{Int}(undef, j - i + 1)
        @simd ivdep for k = i:j
            result[k-i+1] = tree[k] - 1
        end
        return LevelSequence(result)
    end
end


@inline function extract_rooted_leg(tree::LevelSequence, i::Int, j::Int)
    @inbounds begin
        result = Vector{Int}(undef, j - i + 2)
        result[1] = 1
        @simd ivdep for k = i:j
            result[k-i+2] = tree[k]
        end
        return LevelSequence(result)
    end
end


function extract_legs(tree::LevelSequence)
    @inbounds begin
        n = length(tree)
        num_legs = count_legs(tree)
        result = Vector{LevelSequence}(undef, num_legs)
        if iszero(num_legs)
            @assert n == 1
            @assert tree[1] == 1
        else
            @assert n > 1
            @assert tree[1] == 1
            @assert tree[2] == 2
            k = 0
            i = 2
            for j = 3:n
                if tree[j] == 2
                    result[k+=1] = extract_leg(tree, i, j - 1)
                    i = j
                end
            end
            result[k+=1] = extract_leg(tree, i, n)
        end
        return result
    end
end


function extract_rooted_legs(tree::LevelSequence)
    @inbounds begin
        n = length(tree)
        num_legs = count_legs(tree)
        result = Vector{LevelSequence}(undef, num_legs)
        if iszero(num_legs)
            @assert n == 1
            @assert tree[1] == 1
        else
            @assert n > 1
            @assert tree[1] == 1
            @assert tree[2] == 2
            k = 0
            i = 2
            for j = 3:n
                if tree[j] == 2
                    result[k+=1] = extract_rooted_leg(tree, i, j - 1)
                    i = j
                end
            end
            result[k+=1] = extract_rooted_leg(tree, i, n)
        end
        return result
    end
end


###################################################### COMBINING LEVEL SEQUENCES


butcher_product(s::LevelSequence, t::LevelSequence) =
    LevelSequence(append!(copy(s.data), v + 1 for v in t.data))


function butcher_bracket(trees::Vector{LevelSequence})
    @inbounds begin
        result = Vector{Int}(undef, sum(length, trees) + 1)
        result[1] = 1
        k = 1
        for tree in trees
            @simd ivdep for i = 1:length(tree)
                result[k+i] = tree[i] + 1
            end
            k += length(tree)
        end
        return LevelSequence(result)
    end
end


###################################################### ANALYZING LEVEL SEQUENCES


# @inbounds not necessary; bounds check elided.
@inline is_valid(tree::LevelSequence) =
    !isempty(tree.data) && tree[1] == 1 && all(
        2 <= tree[i] <= tree[i-1] + 1 for i = 2:length(tree))


function is_canonical(tree::LevelSequence)
    @assert is_valid(tree)
    legs = extract_legs(tree)
    return issorted(legs; rev=true) && all(is_canonical, legs)
end


# @inbounds not necessary; bounds check elided.
@inline is_tall(tree::LevelSequence) = all(
    tree[i] == i for i = 1:length(tree))
@inline is_bushy(tree::LevelSequence) = all(
    tree[i] == min(i, 2) for i = 1:length(tree))


function canonize(tree::LevelSequence)
    @assert is_valid(tree)
    return issorted(tree.data) ? copy(tree) : butcher_bracket(
        sort!([canonize(leg) for leg in extract_legs(tree)]; rev=true))
end


function is_linear(tree::LevelSequence)
    data = canonize(tree).data
    n = length(data)
    for i = 1:n-1
        if data[i] > data[i+1]
            return issorted(view(data, i+1:n); rev=true)
        end
    end
    return true
end


function classify_verner_type(tree::LevelSequence)
    if is_bushy(tree)
        return :A
    elseif issorted(tree.data)
        return :B
    elseif is_linear(tree)
        return :C
    else
        return :D
    end
end


###################################################### MODIFYING LEVEL SEQUENCES


attach_leaf_left(tree::LevelSequence, index::Int) =
    LevelSequence(insert!(copy(tree.data), index + 1, tree[index] + 1))


function attach_leaf_right(tree::LevelSequence, index::Int)
    value = tree[index] + 1
    next = index + 1
    while next <= length(tree) && tree[next] >= value
        next += 1
    end
    return LevelSequence(insert!(copy(tree.data), next, value))
end


##################################################### GENERATING LEVEL SEQUENCES


function generate_rooted_trees(n::Int; cumulative::Bool)
    if n <= 0
        return LevelSequence[]
    end
    result = frontier = [LevelSequence([1])]
    for _ = 2:n
        next = LevelSequence[]
        seen = Set{LevelSequence}()
        for tree in frontier
            for i = 1:length(tree)
                new_tree = canonize(attach_leaf_left(tree, i))
                if !(new_tree in seen)
                    push!(next, new_tree)
                    push!(seen, new_tree)
                end
            end
        end
        if cumulative
            append!(result, next)
        else
            result = next
        end
        frontier = next
    end
    return result
end


function generate_butcher_trees(n::Int; cumulative::Bool)
    if n <= 0
        return LevelSequence[]
    end
    @inbounds begin
        ranges = Vector{UnitRange{Int}}(undef, n)
        factors = [(NULL_INDEX, NULL_INDEX)]
        ranges[1] = 1:1
        for i = 2:n
            start = length(factors) + 1
            for j = 1:i-1
                for r in ranges[j]
                    for l in ranges[i-j]
                        if l == 1 || r <= factors[l][2]
                            push!(factors, (l, r))
                        end
                    end
                end
            end
            ranges[i] = start:length(factors)
        end
        trees = Vector{LevelSequence}(undef, length(factors))
        trees[1] = LevelSequence([1])
        for i = 2:length(factors)
            left, right = factors[i]
            trees[i] = canonize(butcher_product(trees[left], trees[right]))
        end
        return cumulative ? trees : trees[ranges[n]]
    end
end


################################################### FAST LEVEL SEQUENCE ITERATOR


struct RevLexIterator
    n::Int
end


Base.eltype(::Type{RevLexIterator}) = LevelSequence


function Base.length(iter::RevLexIterator)
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
                    acc = 0
                    for d = 1:j
                        if j % d == 0
                            acc += d * x[d]
                        end
                    end
                    result += acc * x[i-j]
                end
                x[i] = div(result, i - 1)
            end
            return x[n]
        end
    end
end


function Base.iterate(iter::RevLexIterator)
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
    iter::RevLexIterator,
    (L, PREV, SAVE, p)::Tuple{Vector{Int},Vector{Int},Vector{Int},Int},
)
    # This function is based on the GENERATE-NEXT-TREE
    # algorithm from Figure 3 of the following paper:
    # CONSTANT TIME GENERATION OF ROOTED TREES
    # TERRY BEYER AND SANDRA MITCHELL HEDETNIEMI
    # SIAM J. COMPUT. Vol. 9, No. 4, November 1980
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


########################################################## ROOTED TREE INTERFACE


function rooted_trees(n::Int; tree_ordering::Symbol=:reverse_lexicographic)
    if tree_ordering == :reverse_lexicographic
        return collect(RevLexIterator(n))
    elseif tree_ordering == :lexicographic
        return reverse!(collect(RevLexIterator(n)))
    elseif tree_ordering == :attach
        return generate_rooted_trees(n; cumulative=false)
    elseif tree_ordering == :butcher
        return generate_butcher_trees(n; cumulative=false)
    else
        throw(ArgumentError(
            "Unknown tree_ordering $tree_ordering " *
            "(expected :attach, :butcher, " *
            ":lexicographic, or :reverse_lexicographic)"))
    end
end


function all_rooted_trees(n::Int; tree_ordering::Symbol=:reverse_lexicographic)
    if tree_ordering == :reverse_lexicographic
        return reduce(vcat, collect(RevLexIterator(i)) for i = 1:n;
            init=LevelSequence[])
    elseif tree_ordering == :lexicographic
        return reduce(vcat, reverse!(collect(RevLexIterator(i))) for i = 1:n;
            init=LevelSequence[])
    elseif tree_ordering == :attach
        return generate_rooted_trees(n; cumulative=true)
    elseif tree_ordering == :butcher
        return generate_butcher_trees(n; cumulative=true)
    else
        throw(ArgumentError(
            "Unknown tree_ordering $tree_ordering " *
            "(expected :attach, :butcher, " *
            ":lexicographic, or :reverse_lexicographic)"))
    end
end


################################################## COMBINATORICS OF ROOTED TREES


butcher_density(tree::LevelSequence) = reduce(*,
    butcher_density(leg) for leg in extract_legs(tree);
    init=BigInt(length(tree)))


butcher_symmetry(tree::LevelSequence) = reduce(*,
    butcher_symmetry(leg)^count * factorial(BigInt(count))
    for (leg, count) in counts(extract_legs(canonize(tree)));
    init=BigInt(1))


################################################################################


function push_necessary_subtrees!(
    subtrees::Set{LevelSequence},
    tree::LevelSequence,
)
    push!(subtrees, tree)
    if count_legs(tree) > 1
        for rooted_leg in extract_rooted_legs(tree)
            push!(subtrees, rooted_leg)
        end
    end
    for leg in extract_legs(tree)
        push_necessary_subtrees!(subtrees, leg)
    end
    return subtrees
end


function push_necessary_subtrees!(
    subtrees::Set{LevelSequence},
    trees::AbstractVector{LevelSequence},
)
    for tree in trees
        push_necessary_subtrees!(subtrees, tree)
    end
    return subtrees
end


necessary_subtrees(trees::AbstractVector{LevelSequence}) =
    push_necessary_subtrees!(Set{LevelSequence}(), trees)


function has_factors(trees::Set{LevelSequence}, tree::LevelSequence)
    leg_count = count_legs(tree)
    if leg_count == 0
        return true
    elseif leg_count == 1
        return extract_leg(tree, 2, length(tree)) in trees
    elseif leg_count == 2
        left, right = extract_rooted_legs(tree)
        return (left in trees) && (right in trees)
    else
        @assert leg_count > 2
        for (left, right) in BipartitionIterator(extract_rooted_legs(tree))
            if ((butcher_bracket(left) in trees) &&
                (butcher_bracket(right) in trees))
                return true
            end
        end
        return false
    end
end


has_all_factors(trees::Set{LevelSequence}) = all(
    has_factors(trees, tree) for tree in trees)


function best_factor(trees::Set{LevelSequence})
    factors = Dict{LevelSequence,Set{LevelSequence}}()
    for tree in trees
        if !has_factors(trees, tree)
            @assert count_legs(tree) > 2
            for (left, right) in BipartitionIterator(extract_rooted_legs(tree))
                left_factor = butcher_bracket(left)
                right_factor = butcher_bracket(right)
                left_present = left_factor in trees
                right_present = right_factor in trees
                @assert !(left_present && right_present)
                if left_present
                    add_dict!(factors, right_factor, tree)
                elseif right_present
                    add_dict!(factors, left_factor, tree)
                elseif left_factor == right_factor
                    add_dict!(factors, left_factor, tree)
                end
            end
        end
    end
    if isempty(factors)
        return nothing
    else
        return minimum((-length(list), length(tree), tree)
                       for (tree, list) in factors)[3]
    end
end


########################################################### BUTCHER INSTRUCTIONS


struct ButcherInstruction
    left::Int
    right::Int
    depth::Int
    deficit::Int
end


depth(insn::ButcherInstruction) = insn.depth


function find_butcher_instruction(
    instructions::Vector{ButcherInstruction},
    indices::Dict{LevelSequence,Int},
    tree::LevelSequence,
)
    legs = extract_legs(tree)
    if isempty(legs)
        return ButcherInstruction(NULL_INDEX, NULL_INDEX, 1, 0)
    elseif isone(length(legs))
        leg = only(legs)
        if haskey(indices, leg)
            index = indices[leg]
            instruction = instructions[index]
            return ButcherInstruction(index, NULL_INDEX,
                instruction.depth + 1, instruction.deficit + 1)
        else
            return nothing
        end
    else
        candidates = ButcherInstruction[]
        for (left_legs, right_legs) in BipartitionIterator(legs)
            left_tree = butcher_bracket(left_legs)
            if haskey(indices, left_tree)
                right_tree = butcher_bracket(right_legs)
                if haskey(indices, right_tree)
                    left_index = indices[left_tree]
                    right_index = indices[right_tree]
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
    indices::Dict{LevelSequence,Int}, tree::LevelSequence
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


permute_butcher_instruction(insn::ButcherInstruction, perm::Vector{Int}) =
    ButcherInstruction(
        insn.left == NULL_INDEX ? NULL_INDEX : perm[insn.left],
        insn.right == NULL_INDEX ? NULL_INDEX : perm[insn.right],
        insn.depth, insn.deficit)


function glex_order(a::LevelSequence, b::LevelSequence)
    len_a = length(a)
    len_b = length(b)
    if len_a < len_b
        return true
    elseif len_a > len_b
        return false
    else
        return a < b
    end
end


function grevlex_order(a::LevelSequence, b::LevelSequence)
    len_a = length(a)
    len_b = length(b)
    if len_a < len_b
        return true
    elseif len_a > len_b
        return false
    else
        return a > b
    end
end


function build_instructions(
    trees::Vector{LevelSequence};
    optimize::Bool,
    sort_by_depth::Bool,
)
    @assert allunique(trees)
    instructions = ButcherInstruction[]
    indices = Dict{LevelSequence,Int}()
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
        if instruction.left == NULL_INDEX
            @assert instruction.right == NULL_INDEX
            push!(result, LevelSequence([1]))
        elseif instruction.right == NULL_INDEX
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
    selected_indices::Vector{Int}
    source_indices::Vector{Int}
    extension_indices::Vector{Int}
    rooted_sum_indices::Vector{Pair{Int,Int}}
    rooted_sum_ranges::Vector{UnitRange{Int}}
end


function compute_reverse_relationships(
    instructions::Vector{ButcherInstruction}
)
    extensions = [NULL_INDEX for _ in instructions]
    rooted_sums = [Pair{Int,Int}[] for _ in instructions]
    for (i, instruction) in pairs(instructions)
        if instruction.left == NULL_INDEX
            @assert instruction.right == NULL_INDEX
        elseif instruction.right == NULL_INDEX
            extensions[instruction.left] = i
        else
            push!(rooted_sums[instruction.left], instruction.right => i)
            push!(rooted_sums[instruction.right], instruction.left => i)
        end
    end
    return (extensions, rooted_sums)
end


function ButcherInstructionTable(
    trees::Vector{LevelSequence};
    optimize::Bool,
    sort_by_depth::Bool,
)
    instructions, selected_indices = build_instructions(trees;
        optimize=optimize, sort_by_depth=sort_by_depth)
    source_indices = [NULL_INDEX for _ in instructions]
    for (i, j) in pairs(selected_indices)
        source_indices[j] = i
    end
    extension_indices, rooted_sum_lists =
        compute_reverse_relationships(instructions)
    rooted_sum_indices = reduce(vcat, rooted_sum_lists; init=Pair{Int,Int}[])
    end_indices = cumsum(length.(rooted_sum_lists))
    start_indices = vcat([1], end_indices[1:end-1] .+ 1)
    rooted_sum_ranges = UnitRange{Int}.(start_indices, end_indices)
    return ButcherInstructionTable(
        instructions, selected_indices, source_indices,
        extension_indices, rooted_sum_indices, rooted_sum_ranges)
end


############################################################### GRAPHICAL OUTPUT


svg_string(x::Float64) = rstrip(rstrip(@sprintf("%.15f", x), '0'), '.')


function svg_string(
    x_min::Float64,
    x_max::Float64,
    y_min::Float64,
    y_max::Float64,
)
    width_string = svg_string(x_max - x_min)
    height_string = svg_string(y_max - y_min)
    return @sprintf(
        """<svg xmlns="http://www.w3.org/2000/svg" width="%s" height="%s" viewBox="%s %s %s %s">""",
        width_string, height_string,
        svg_string(x_min), svg_string(y_min),
        width_string, height_string)
end


struct Circle
    cx::Float64
    cy::Float64
    r::Float64
end


min_x(circle::Circle) = circle.cx - circle.r
min_y(circle::Circle) = circle.cy - circle.r
max_x(circle::Circle) = circle.cx + circle.r
max_y(circle::Circle) = circle.cy + circle.r


svg_string(circle::Circle, color::String) = @sprintf(
    """<circle cx="%s" cy="%s" r="%s" stroke="%s" />""",
    svg_string(circle.cx), svg_string(circle.cy),
    svg_string(circle.r), color)


struct Line
    x1::Float64
    y1::Float64
    x2::Float64
    y2::Float64
end


min_x(line::Line) = min(line.x1, line.x2)
min_y(line::Line) = min(line.y1, line.y2)
max_x(line::Line) = max(line.x1, line.x2)
max_y(line::Line) = max(line.y1, line.y2)


svg_string(line::Line, color::String, width::Float64) = @sprintf(
    """<line x1="%s" y1="%s" x2="%s" y2="%s" stroke="%s" stroke-width="%s" />""",
    svg_string(line.x1), svg_string(line.y1),
    svg_string(line.x2), svg_string(line.y2),
    color, width)


function tree_width!(widths::Dict{LevelSequence,Int}, tree::LevelSequence)
    if haskey(widths, tree)
        return widths[tree]
    else
        legs = extract_legs(tree)
        if isempty(legs)
            return 0
        else
            result = sum(tree_width!(widths, leg)
                         for leg in legs) + (length(legs) - 1)
            widths[tree] = result
            return result
        end
    end
end


function tree_diagram!(
    circles::Vector{Circle},
    lines::Vector{Line},
    widths::Dict{LevelSequence,Int},
    tree::LevelSequence,
    x::Float64,
    y::Float64,
    radius::Float64,
    spacing::Float64,
)
    push!(circles, Circle(x, y, radius))
    legs = extract_legs(tree)
    if !isempty(legs)
        total_width = tree_width!(widths, tree) * spacing
        leg_x = x - 0.5 * total_width
        leg_y = y - spacing
        for leg in legs
            leg_width = tree_width!(widths, leg) * spacing
            leg_x += 0.5 * leg_width
            push!(lines, Line(x, y, leg_x, leg_y))
            tree_diagram!(circles, lines, widths, leg,
                leg_x, leg_y, radius, spacing)
            leg_x += 0.5 * leg_width + spacing
        end
    end
    return (circles, lines)
end


tree_diagram(tree::LevelSequence, radius::Float64, spacing::Float64) =
    tree_diagram!(Circle[], Line[], Dict{LevelSequence,Int}(),
        tree, 0.0, 0.0, radius, spacing)


function svg_string(
    tree::LevelSequence,
    radius::Float64,
    spacing::Float64,
    line_width::Float64,
)
    circles, lines = tree_diagram(tree, radius, spacing)
    io = IOBuffer()
    println(io, svg_string(
        minimum(min_x, circles) - 0.5 * spacing,
        maximum(max_x, circles) + 0.5 * spacing,
        minimum(min_y, circles) - 0.5 * spacing,
        maximum(max_y, circles) + 0.5 * spacing))
    for line in lines
        println(io, svg_string(line, "white", line_width + 1.0))
        println(io, svg_string(line, "black", line_width - 1.0))
    end
    for circle in circles
        println(io, svg_string(circle, "white"))
    end
    print(io, "</svg>")
    return String(take!(io))
end


function Base.show(io::IO, ::MIME"text/html", tree::LevelSequence)
    println(io, "<div>")
    println(io, svg_string(tree, 7.0, 25.0, 4.0))
    println(io, "</div>")
end


function Base.show(io::IO, ::MIME"text/html",
    trees::AbstractVector{LevelSequence})
    println(io, "<div>")
    for tree in trees
        println(io, svg_string(tree, 7.0, 25.0, 4.0))
    end
    println(io, "</div>")
end


end # module ButcherInstructions
