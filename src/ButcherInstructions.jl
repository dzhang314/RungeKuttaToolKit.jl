module ButcherInstructions


using Printf
using ..RungeKuttaToolKit: NULL_INDEX


################################################################### BIPARTITIONS


struct BipartitionIterator{T}
    items::Vector{T}
end


@inline Base.eltype(::Type{BipartitionIterator{T}}) where {T} =
    Tuple{Vector{T},Vector{T}}


@inline Base.length(iter::BipartitionIterator{T}) where {T} =
    isempty(iter.items) ? 0 : ((1 << (length(iter.items) - 1)) - 1)


function Base.iterate(iter::BipartitionIterator{T}) where {T}
    n = length(iter.items)
    return (n < 2) ? nothing : (
        ((T[@inbounds iter.items[1]], @inbounds iter.items[2:n])),
        (one(UInt) << (n - 1)) - 1)
end


function Base.iterate(iter::BipartitionIterator{T}, state::UInt) where {T}
    if state < 2
        return nothing
    end
    n = length(iter.items)
    state -= 1
    next_state = state
    left = T[@inbounds iter.items[1]]
    right = T[]
    @inbounds for i = 2:n
        push!(ifelse(iszero(state & 1), left, right), iter.items[i])
        state >>= 1
    end
    return ((left, right), next_state)
end


################################################################ LEVEL SEQUENCES


struct LevelSequence
    data::Vector{Int}
end


@inline Base.length(s::LevelSequence) = length(s.data)
@inline Base.isempty(s::LevelSequence) = isempty(s.data)
@inline Base.getindex(s::LevelSequence, i::Int) = s.data[i]
@inline Base.getindex(s::LevelSequence, r::AbstractRange) =
    LevelSequence(s.data[r])
@inline Base.copy(s::LevelSequence) = LevelSequence(copy(s.data))

@inline Base.:(==)(s::LevelSequence, t::LevelSequence) = (s.data == t.data)
@inline Base.hash(s::LevelSequence, h::UInt) = hash(s.data, h)
@inline Base.isless(s::LevelSequence, t::LevelSequence) =
    isless(s.data, t.data)


function decrement!(s::LevelSequence)
    @simd ivdep for i = 1:length(s)
        @inbounds s.data[i] -= 1
    end
    return s
end


function increment!(s::LevelSequence)
    @simd ivdep for i = 1:length(s)
        @inbounds s.data[i] += 1
    end
    return s
end


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
                    result[i] = decrement!(s[last:j-1])
                    i += 1
                    last = j
                end
            end
            result[i] = decrement!(s[last:n])
        end
        return result
    end
end


function is_canonical(s::LevelSequence)
    legs = extract_legs(s)
    if !issorted(legs; rev=true)
        return false
    end
    for leg in legs
        if !is_canonical(leg)
            return false
        end
    end
    return true
end


function is_tall(s::LevelSequence)
    @inbounds for i = 1:length(s)
        if s[i] != i
            return false
        end
    end
    return true
end


function is_bushy(s::LevelSequence)
    @assert !isempty(s)
    @assert isone(s[1])
    @inbounds for i = 2:length(s)
        if s[i] != 2
            return false
        end
    end
    return true
end


function canonize(s::LevelSequence)
    if is_tall(s) || is_bushy(s)
        return copy(s)
    end
    result = [1]
    for leg in sort!([canonize(leg) for leg in extract_legs(tree)]; rev=true)
        for vertex in leg
            push!(result, vertex + 1)
        end
    end
    return LevelSequence(result)
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
    reduce(vcat, rooted_trees(i; tree_ordering=tree_ordering) for i = 1:n;
        init=LevelSequence[])


################################################## COMBINATORICS OF ROOTED TREES


function butcher_density(tree::LevelSequence)
    result = BigInt(length(tree))
    for leg in extract_legs(tree)
        result *= butcher_density(leg)
    end
    return result
end


function butcher_symmetry(tree::LevelSequence)
    @assert is_canonical(tree)
    leg_counts = LevelSequenceDict()
    for leg in extract_legs(tree)
        if haskey(leg_counts, leg)
            leg_counts[leg] += 1
        else
            leg_counts[leg] = 1
        end
    end
    result = BigInt(1)
    for (data, count) in leg_counts.entries
        leg_symmetry = butcher_symmetry(LevelSequence(data))
        result *= leg_symmetry * factorial(BigInt(count))
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
        for (left, right) in BipartitionIterator(legs)
            left_leg = LevelSequence(reduce(vcat,
                (leg.data .+ 1 for leg in left); init=[1]))
            if haskey(indices, left_leg)
                right_leg = LevelSequence(reduce(vcat,
                    (leg.data .+ 1 for leg in right); init=[1]))
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
    insn::ButcherInstruction, perm::Vector{Int}
) = ButcherInstruction(
    (insn.left == NULL_INDEX) ? NULL_INDEX : perm[insn.left],
    (insn.right == NULL_INDEX) ? NULL_INDEX : perm[insn.right],
    insn.depth, insn.deficit)


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
        return a > b
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
    optimize::Bool,
    sort_by_depth::Bool,
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
        """\
        <svg xmlns="http://www.w3.org/2000/svg" \
        width="%s" height="%s" viewBox="%s %s %s %s">\
        """,
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
    "<circle cx=\"%s\" cy=\"%s\" r=\"%s\" stroke=\"%s\" />",
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
    """\
    <line x1="%s" y1="%s" x2="%s" y2="%s" \
    stroke="%s" stroke-width="%s" />\
    """,
    svg_string(line.x1), svg_string(line.y1),
    svg_string(line.x2), svg_string(line.y2),
    color, width)


function tree_width!(widths::LevelSequenceDict, tree::LevelSequence)
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
    widths::LevelSequenceDict,
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
    tree_diagram!(Circle[], Line[], LevelSequenceDict(),
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
