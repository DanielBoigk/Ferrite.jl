# Newest vertex bisection (NVB) of linear triangle meshes [Mitchell1991](@cite),
# [Stevenson2008](@cite): conforming local refinement without hanging nodes.
#
# Every triangle is stored as (i, j, k), counter-clockwise, with refinement edge (i, j) and
# newest vertex k. Bisection inserts the midpoint m of (i, j) and creates the children
# (k, i, m) and (j, k, m): both counter-clockwise, with newest vertex m and refinement edges
# (k, i) and (j, k). The initial refinement edge of every triangle is its longest edge.
#
# Conformity: an edge that has been split (its midpoint exists) must not remain in any leaf.
# After the marked cells are bisected, the closure bisects every leaf that still contains a
# split edge. A leaf whose split edge is not its refinement edge is bisected along its
# refinement edge first; the child that inherits the split edge has it as its refinement
# edge and is bisected next. Only the cells around a newly split edge can become
# non-conforming, so they are found through a map from edges to the leaves containing them,
# and a refinement costs time proportional to the number of bisections, not to the size of
# the mesh.

"""
    BisectionMesh(grid::AbstractGrid, maxlevel::Integer = 30)

Builds an adaptively refinable mesh from a `grid` of linear `Triangle`s. It is refined by
newest vertex bisection: every refinement of a cell halves it, and neighbouring cells are
bisected as well until the mesh is conforming again. Unlike [`ForestBWG`](@ref
Ferrite.AMR.ForestBWG), the refined grids therefore have no hanging nodes and need no
conformity constraints, so every interpolation can be used on them.

`maxlevel` is the maximum number of bisections of a cell of `grid`; two bisections quarter a
cell. Marked cells at this level are not refined, but cells may exceed it by the bisections
that restore conformity.

Refine with [`refine!`](@ref Ferrite.AMR.refine!) and get the current grid with
[`creategrid`](@ref Ferrite.AMR.creategrid). The `cellsets` and `facetsets` of `grid` are
carried over to the refined grids. Coarsening is not implemented.

The refinement level of every cell of the current grid (the number of bisections from its
cell in `grid`) is stored in the field `levels`.
"""
mutable struct BisectionMesh{T}
    nodes::Vector{Vec{2, T}}
    tris::Vector{NTuple{3, Int}}                      # leaves (i, j, k)
    levels::Vector{Int}                               # bisections from the initial cell
    cellsets::Vector{Vector{String}}                  # cell sets of every leaf
    facetsets::Dict{String, Set{Tuple{Int, Int}}}     # facet sets as sorted node pairs
    midpoints::Dict{Tuple{Int, Int}, Int}             # split edge → midpoint node
    edgecells::Dict{Tuple{Int, Int}, Vector{Int}}     # edge → leaves containing it
    maxlevel::Int
end

_signed_area(p, q, r) = ((q - p)[1] * (r - p)[2] - (q - p)[2] * (r - p)[1]) / 2

_edges((i, j, k)) = (minmax(i, j), minmax(j, k), minmax(k, i))

function BisectionMesh(grid::Ferrite.AbstractGrid, maxlevel::Integer = 30)
    getcelltype(grid) === Triangle ||
        throw(ArgumentError("newest vertex bisection needs a grid of linear triangles, got $(getcelltype(grid))"))
    maxlevel >= 0 || throw(DomainError(maxlevel, "maxlevel must be non-negative"))
    nodes = [get_node_coordinate(grid, i) for i in 1:getnnodes(grid)]
    tris = NTuple{3, Int}[]
    for cell in getcells(grid)
        a, b, c = cell.nodes
        _signed_area(nodes[a], nodes[b], nodes[c]) < 0 && ((b, c) = (c, b))
        t = (a, b, c)
        # rotate the longest edge to (i, j)
        r = argmax(ntuple(s -> norm(nodes[t[mod1(s + 1, 3)]] - nodes[t[s]]), 3))
        push!(tris, (t[r], t[mod1(r + 1, 3)], t[mod1(r + 2, 3)]))
    end
    names = [String[] for _ in tris]
    for (name, set) in Ferrite.getcellsets(grid), c in set
        push!(names[c], name)
    end
    facetsets = Dict{String, Set{Tuple{Int, Int}}}()
    for (name, set) in Ferrite.getfacetsets(grid)
        facetsets[name] = Set(minmax(Ferrite.facets(getcells(grid, fi.idx[1]))[fi.idx[2]]...) for fi in set)
    end
    edgecells = Dict{Tuple{Int, Int}, Vector{Int}}()
    for (c, t) in enumerate(tris), e in _edges(t)
        push!(get!(edgecells, e, Int[]), c)
    end
    return BisectionMesh(
        nodes, tris, zeros(Int, length(tris)), names, facetsets,
        Dict{Tuple{Int, Int}, Int}(), edgecells, Int(maxlevel)
    )
end

Ferrite.getncells(m::BisectionMesh) = length(m.tris)

function _replace_edgecell!(m::BisectionMesh, e, old, new)
    cells = m.edgecells[e]
    cells[findfirst(==(old), cells)] = new
    return
end

_has_split_edge(m::BisectionMesh, t) = any(e -> haskey(m.midpoints, e), _edges(m.tris[t]))

"""
    _bisect!(mesh::BisectionMesh, t::Int, dirty::Vector{Int})

Bisect leaf `t` of `mesh` along its refinement edge `(i, j)`. The leaf `t` becomes the child
`(k, i, m)` and the child `(j, k, m)` is appended, with `m` the midpoint of `(i, j)`
(created unless a neighbour has split the edge before). Updates the edge-to-leaf map, the
facet sets and the levels, and pushes to `dirty` every leaf that may now contain a split
edge: the neighbour across a newly split edge, and both children.
"""
function _bisect!(m::BisectionMesh, t::Int, dirty::Vector{Int})
    i, j, k = m.tris[t]
    e = minmax(i, j)
    mid = get(m.midpoints, e, 0)
    if mid == 0
        push!(m.nodes, (m.nodes[i] + m.nodes[j]) / 2)
        mid = length(m.nodes)
        m.midpoints[e] = mid
        for set in values(m.facetsets)
            if e in set
                delete!(set, e)
                push!(set, minmax(e[1], mid), minmax(mid, e[2]))
            end
        end
        # the other leaf on the split edge has to be bisected as well
        for c in m.edgecells[e]
            c == t || push!(dirty, c)
        end
    end
    n = length(m.tris) + 1
    # edges: (i, j) is replaced by (i, m) and (m, j); (j, k) moves to the new child n;
    # (k, i) stays with t; (k, m) is shared by both children
    cells = m.edgecells[e]
    deleteat!(cells, findfirst(==(t), cells))
    isempty(cells) && delete!(m.edgecells, e)
    _replace_edgecell!(m, minmax(j, k), t, n)
    push!(get!(m.edgecells, minmax(i, mid), Int[]), t)
    push!(get!(m.edgecells, minmax(mid, j), Int[]), n)
    push!(get!(m.edgecells, minmax(k, mid), Int[]), t, n)
    l = m.levels[t] + 1
    m.tris[t] = (k, i, mid)
    push!(m.tris, (j, k, mid))
    m.levels[t] = l
    push!(m.levels, l)
    push!(m.cellsets, m.cellsets[t])
    # a child that kept an edge split earlier (by a neighbour) is not conforming yet
    push!(dirty, t, n)
    return
end

"""
    refine!(mesh::BisectionMesh, cellids::AbstractVector{<:Integer})
    refine!(mesh::BisectionMesh, cellid::Integer)

Bisect the cells `cellids` of the current grid ([`creategrid`](@ref Ferrite.AMR.creategrid))
once, then bisect neighbouring cells until the mesh is conforming. Duplicates are ignored, and
cells at the maximum level are skipped. Cell ids refer to the grid before the call: a
bisected cell keeps its id for one of its children, and new cells are appended.
"""
function refine!(m::BisectionMesh, cellids::AbstractVector{<:Integer})
    dirty = Int[]
    for t in unique(cellids)
        1 <= t <= length(m.tris) || throw(BoundsError(m.tris, t))
        m.levels[t] < m.maxlevel && _bisect!(m, Int(t), dirty)
    end
    while !isempty(dirty)
        t = pop!(dirty)
        _has_split_edge(m, t) && _bisect!(m, t, dirty)
    end
    return m
end
refine!(m::BisectionMesh, cellid::Integer) = refine!(m, [cellid])

"""
    creategrid(mesh::BisectionMesh) -> Grid

The current grid of a [`BisectionMesh`](@ref Ferrite.AMR.BisectionMesh): a conforming
`Grid` of `Triangle`s with the `cellsets` and `facetsets` of the initial grid. Its cells are
numbered as in `mesh`, and the nodes of the initial grid keep their numbers.

!!! warning "Only `facetsets` and `cellsets` are transferred"
    As for [`ForestBWG`](@ref Ferrite.AMR.ForestBWG), the `vertexsets` and `nodesets` of
    the initial grid are not carried over.
"""
function creategrid(m::BisectionMesh)
    cells = [Triangle(t) for t in m.tris]
    nodes = [Node(x) for x in m.nodes]
    edge_names = Dict{Tuple{Int, Int}, Vector{String}}()
    for (name, set) in m.facetsets, e in set
        push!(get!(edge_names, e, String[]), name)
    end
    facetsets = Dict{String, OrderedSet{FacetIndex}}(n => OrderedSet{FacetIndex}() for n in keys(m.facetsets))
    for (c, t) in enumerate(m.tris), (lf, (a, b)) in enumerate(((t[1], t[2]), (t[2], t[3]), (t[3], t[1])))
        for name in get(edge_names, minmax(a, b), ())
            push!(facetsets[name], FacetIndex(c, lf))
        end
    end
    cellsets = Dict{String, OrderedSet{Int}}()
    for (c, names) in enumerate(m.cellsets), n in names
        push!(get!(cellsets, n, OrderedSet{Int}()), c)
    end
    return Grid(cells, nodes; facetsets, cellsets)
end
