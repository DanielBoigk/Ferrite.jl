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
carried over to the refined grids. [`coarsen!`](@ref Ferrite.AMR.coarsen!) undoes
bisections, exactly inverting the refinement.

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

# Coarsening [ChenZhang2010](@cite): the exact inverse of bisection, found by a local test on
# the vertices, without a refinement tree. A vertex v created by bisecting the edge (i, j) is
# removable if it is the newest vertex of every leaf containing it (its star) and these leaves
# pair up into the children T₁ = (k, i, v), T₂ = (j, k, v) of parents (i, j, k) =
# (T₁[2], T₂[1], T₁[1]) whose refinement edge is (i, j): two leaves on a boundary edge, four
# on an interior one. The stars of different vertices are disjoint (a leaf has one newest
# vertex), so all removable vertices are merged in one sweep, which undoes the latest
# bisection around each of them.

# Parent merges `(T₁, T₂, parent)` and removed vertices for the marked cells.
function _coarsening_plan(m::BisectionMesh, cellids, require_all_siblings::Bool)
    ncells = length(m.tris)
    for c in cellids
        1 <= c <= ncells || throw(BoundsError(m.tris, c))
    end
    merges = Tuple{Int, Int, NTuple{3, Int}}[]
    removed = Int[]
    isempty(cellids) && return merges, removed
    parentedge = Dict{Int, Tuple{Int, Int}}(v => e for (e, v) in m.midpoints)
    stars = Dict{Int, Vector{Int}}()
    for c in cellids
        v = m.tris[c][3]
        haskey(parentedge, v) && (stars[v] = Int[])
    end
    for (c, t) in enumerate(m.tris), v in t
        haskey(stars, v) && push!(stars[v], c)
    end
    marked = falses(ncells)
    marked[cellids] .= true
    for (v, star) in stars
        length(star) in (2, 4) || continue
        all(c -> m.tris[c][3] == v, star) || continue
        require_all_siblings && !all(c -> marked[c], star) && continue
        pairs = Tuple{Int, Int, NTuple{3, Int}}[]
        for a in star, b in star
            ta, tb = m.tris[a], m.tris[b]
            # a = (k, i, v) and b = (j, k, v), children of (i, j, k) with refinement edge (i, j)
            if ta[1] == tb[2] && minmax(ta[2], tb[1]) == parentedge[v]
                push!(pairs, (a, b, (ta[2], tb[1], ta[1])))
            end
        end
        length(pairs) == length(star) ÷ 2 || continue
        allunique(c for (a, b, _) in pairs for c in (a, b)) || continue
        append!(merges, pairs)
        push!(removed, v)
    end
    return merges, removed
end

# Merge, then renumber cells and nodes. Returns the map from old to new cell ids (a merged
# child maps to its parent).
function _apply_coarsening!(m::BisectionMesh, merges, removed)
    ncells = length(m.tris)
    isempty(merges) && return collect(1:ncells)
    parentedge = Dict{Int, Tuple{Int, Int}}(m.midpoints[e] => e for e in (minmax(t[1], t[2]) for (_, _, t) in merges))
    deleted = falses(ncells)
    parentof = collect(1:ncells)
    for (a, b, parent) in merges
        m.tris[a] = parent
        m.levels[a] -= 1
        deleted[b] = true
        parentof[b] = a
    end
    for v in removed
        i, j = e = parentedge[v]
        delete!(m.midpoints, e)
        for set in values(m.facetsets)
            if minmax(i, v) in set || minmax(v, j) in set
                delete!(set, minmax(i, v))
                delete!(set, minmax(v, j))
                push!(set, e)
            end
        end
    end
    # cells: drop the merged second children
    newid = cumsum(.!deleted)
    cellmap = [newid[parentof[c]] for c in 1:ncells]
    keep = findall(!, deleted)
    # nodes: drop the removed vertices (all created by bisection, so the nodes of the initial
    # grid keep their numbers)
    gone = falses(length(m.nodes))
    gone[removed] .= true
    nodemap = cumsum(.!gone)
    renumber(n) = nodemap[n]
    m.nodes = m.nodes[.!gone]
    m.tris = [renumber.(t) for t in m.tris[keep]]
    m.levels = m.levels[keep]
    m.cellsets = m.cellsets[keep]
    m.midpoints = Dict{Tuple{Int, Int}, Int}(minmax(renumber.(e)...) => renumber(v) for (e, v) in m.midpoints)
    for (name, set) in m.facetsets
        m.facetsets[name] = Set{Tuple{Int, Int}}(minmax(renumber.(e)...) for e in set)
    end
    m.edgecells = Dict{Tuple{Int, Int}, Vector{Int}}()
    for (c, t) in enumerate(m.tris), e in _edges(t)
        push!(get!(m.edgecells, e, Int[]), c)
    end
    return cellmap
end

"""
    coarsen!(mesh::BisectionMesh, cellids::AbstractVector{<:Integer}; require_all_siblings::Bool = true)

Undo bisections around the cells `cellids` of the current grid, the inverse of
[`refine!`](@ref Ferrite.AMR.refine!). A vertex created by bisection is removed, and the cells
around it are merged back into their parents, if it is the newest vertex of every cell around
it: two cells on the boundary, four in the interior. Coarsening is therefore the exact inverse
of refinement and keeps the mesh conforming. One call undoes at most one bisection around each
vertex, and vertices of the initial grid are never removed.

`require_all_siblings` selects the trigger policy, as for [`ForestBWG`](@ref Ferrite.AMR.ForestBWG):
- `true` (default): a vertex is removed only if **all** cells around it are in `cellids`.
- `false`: a single marked cell removes its newest vertex if possible.

Cells and nodes are renumbered: the remaining cells keep their order, and a merged parent takes
the place of its first child. Runs in `O(n)` for `n` cells.
"""
function coarsen!(m::BisectionMesh, cellids::AbstractVector{<:Integer}; require_all_siblings::Bool = true)
    merges, removed = _coarsening_plan(m, cellids, require_all_siblings)
    _apply_coarsening!(m, merges, removed)
    return m
end

"""
    refine_and_coarsen!(mesh::BisectionMesh, coarsen_ids, refine_ids; require_all_siblings = true)

Coarsen around the cells `coarsen_ids` and refine the cells `refine_ids` of the current grid.
Both id sets refer to the same numbering: [`coarsen!`](@ref Ferrite.AMR.coarsen!) renumbers the
cells, so calling it and [`refine!`](@ref Ferrite.AMR.refine!) back to back with ids from one grid would
misfire. The two id sets must be disjoint, and no cell of `refine_ids` may be merged by the
coarsening; either conflict throws an `ArgumentError` before the mesh is changed.
"""
function refine_and_coarsen!(
        m::BisectionMesh, coarsen_ids::AbstractVector{<:Integer}, refine_ids::AbstractVector{<:Integer};
        require_all_siblings::Bool = true
    )
    isdisjoint(coarsen_ids, refine_ids) ||
        throw(ArgumentError("a cell cannot be marked for both refinement and coarsening"))
    for c in refine_ids
        1 <= c <= length(m.tris) || throw(BoundsError(m.tris, c))
    end
    merges, removed = _coarsening_plan(m, coarsen_ids, require_all_siblings)
    merged = Set(c for (a, b, _) in merges for c in (a, b))
    any(in(merged), refine_ids) &&
        throw(ArgumentError("a cell marked for refinement is merged by the coarsening"))
    cellmap = _apply_coarsening!(m, merges, removed)
    return refine!(m, [cellmap[c] for c in refine_ids])
end
