# Newest vertex bisection of triangle meshes (`BisectionMesh`): conformity, orientation,
# shape regularity, levels and the level cap, transfer of facet and cell sets, and finite
# elements on the refined grids. Mirrors `src/Adaptivity/bisection.jl`.
using Ferrite, Test
using LinearAlgebra, SparseArrays, Random

# Every edge is used by at most two triangles, and no node lies inside an edge that is used
# only once (which is what a hanging node would leave behind).
function is_conforming(grid)
    count = Dict{Tuple{Int, Int}, Int}()
    for c in getcells(grid), (a, b) in Ferrite.facets(c)
        k = minmax(a, b)
        count[k] = get(count, k, 0) + 1
    end
    all(<=(2), values(count)) || return false
    for (k, n) in count
        n == 1 || continue
        pa, pb = get_node_coordinate(grid, k[1]), get_node_coordinate(grid, k[2])
        for v in 1:getnnodes(grid)
            v in k && continue
            pv = get_node_coordinate(grid, v)
            t = (pv - pa) ⋅ (pb - pa) / ((pb - pa) ⋅ (pb - pa))
            0 < t < 1 && norm(pa + t * (pb - pa) - pv) < 1.0e-12 && return false
        end
    end
    return true
end

function signed_area(grid, c)
    x = [get_node_coordinate(grid, n) for n in getcells(grid, c).nodes]
    return ((x[2] - x[1])[1] * (x[3] - x[1])[2] - (x[2] - x[1])[2] * (x[3] - x[1])[1]) / 2
end

function min_angle(grid)
    m = float(π)
    for c in getcells(grid)
        x = [get_node_coordinate(grid, n) for n in c.nodes]
        for i in 1:3
            a, b = x[mod1(i + 1, 3)] - x[i], x[mod1(i + 2, 3)] - x[i]
            m = min(m, acos(clamp(a ⋅ b / (norm(a) * norm(b)), -1, 1)))
        end
    end
    return m
end

function facetset_length(grid, set)
    return sum(set; init = 0.0) do fi
        a, b = Ferrite.facets(getcells(grid, fi.idx[1]))[fi.idx[2]]
        norm(get_node_coordinate(grid, b) - get_node_coordinate(grid, a))
    end
end

@testset "BisectionMesh" begin
    rng = MersenneTwister(8)
    base = generate_grid(Triangle, (4, 4))   # [-1, 1]², 32 cells

    @testset "local refinement is conforming, oriented and area preserving" begin
        mesh = BisectionMesh(base)
        @test getncells(creategrid(mesh)) == 32
        refine!(mesh, 5)
        grid = creategrid(mesh)
        @test grid isa Grid{2, Triangle}
        @test getncells(grid) > 32
        @test is_conforming(grid)
        @test all(c -> signed_area(grid, c) > 0, 1:getncells(grid))
        @test sum(c -> signed_area(grid, c), 1:getncells(grid)) ≈ 4.0
        # the nodes of the initial grid keep their numbers
        @test all(n -> get_node_coordinate(grid, n) == get_node_coordinate(base, n), 1:getnnodes(base))
        # many random refinements: still conforming, and the angles stay bounded below
        # (bisection produces finitely many similarity classes)
        α0 = min_angle(base)
        for _ in 1:12
            refine!(mesh, rand(rng, 1:getncells(mesh), 4))
        end
        grid = creategrid(mesh)
        @test is_conforming(grid)
        @test all(c -> signed_area(grid, c) > 0, 1:getncells(grid))
        @test sum(c -> signed_area(grid, c), 1:getncells(grid)) ≈ 4.0
        @test min_angle(grid) >= α0 / 2 - 1.0e-12
    end

    @testset "uniform refinement, levels and the level cap" begin
        mesh = BisectionMesh(base, 4)
        refine!(mesh, 1:getncells(mesh))           # every triangle bisected once
        @test getncells(mesh) == 64
        refine!(mesh, 1:getncells(mesh))
        @test getncells(mesh) == 128
        @test all(==(2), mesh.levels)
        grid = creategrid(mesh)
        @test all(c -> signed_area(grid, c) ≈ 4 / 128, 1:getncells(grid))
        for _ in 1:4
            refine!(mesh, [1, 1])                  # duplicates are ignored
        end
        @test mesh.levels[1] == 4                   # cells at the cap are not refined
        @test is_conforming(creategrid(mesh))
        n = getncells(mesh)
        refine!(mesh, Int[])
        @test getncells(mesh) == n
    end

    @testset "facet and cell sets follow the refinement" begin
        mesh = BisectionMesh(base)
        for _ in 1:5
            refine!(mesh, rand(rng, 1:getncells(mesh), 6))
        end
        grid = creategrid(mesh)
        for name in ("left", "right", "top", "bottom")
            @test facetset_length(grid, getfacetset(grid, name)) ≈ 2.0
        end
        # the boundary sets are exactly the facets of one cell
        topology = ExclusiveTopology(grid)
        boundary = Set(FacetIndex(c, f) for c in 1:getncells(grid), f in 1:3 if isempty(getneighborhood(topology, grid, FacetIndex(c, f))))
        @test boundary == union((Set(getfacetset(grid, n)) for n in ("left", "right", "top", "bottom"))...)

        cellset_grid = generate_grid(Triangle, (2, 2))
        addcellset!(cellset_grid, "left half", x -> x[1] <= 0)
        mesh = BisectionMesh(cellset_grid)
        refine!(mesh, collect(getcellset(cellset_grid, "left half")))
        grid = creategrid(mesh)
        @test sum(c -> signed_area(grid, c), getcellset(grid, "left half")) ≈ 2.0
    end

    @testset "finite elements on the refined grid" begin
        mesh = BisectionMesh(generate_grid(Triangle, (8, 8)))
        refine!(mesh, [3, 17, 30])
        refine!(mesh, [1, 2])
        grid = creategrid(mesh)
        # no hanging nodes: a quadratic space without constraints reproduces |∇x₁|² over Ω
        ip = Lagrange{RefTriangle, 2}()
        dh = close!(add!(DofHandler(grid), :u, ip))
        cv = CellValues(QuadratureRule{RefTriangle}(2), ip)
        K = allocate_matrix(dh)
        assembler = start_assemble(K)
        Ke = zeros(getnbasefunctions(cv), getnbasefunctions(cv))
        for cell in CellIterator(dh)
            reinit!(cv, cell)
            fill!(Ke, 0)
            for q in 1:getnquadpoints(cv), i in 1:getnbasefunctions(cv), j in 1:getnbasefunctions(cv)
                Ke[i, j] += shape_gradient(cv, q, i) ⋅ shape_gradient(cv, q, j) * getdetJdV(cv, q)
            end
            assemble!(assembler, celldofs(cell), Ke)
        end
        x1 = zeros(ndofs(dh))
        apply_analytical!(x1, dh, :u, x -> x[1])
        @test x1' * K * x1 ≈ 4.0
    end

    # the mesh as a set of triangles in node coordinates, with refinement edges and levels
    function shape(mesh)
        x(n) = Tuple(mesh.nodes[n])
        return Set((x(t[1]), x(t[2]), x(t[3]), l) for (t, l) in zip(mesh.tris, mesh.levels))
    end

    @testset "coarsening inverts refinement" begin
        mesh = BisectionMesh(base)
        before = shape(mesh)
        n0 = getncells(mesh)
        refine!(mesh, 5)
        refine!(mesh, 1:getncells(mesh))
        # every sweep removes the newest vertices; repeat until nothing changes (the closure
        # can bisect a cell twice in one refinement, so this may take more than two sweeps)
        sweeps = 0
        while true
            n = getncells(mesh)
            coarsen!(mesh, 1:n)
            getncells(mesh) == n && break
            sweeps += 1
        end
        @test 2 <= sweeps <= 4
        @test getncells(mesh) == n0
        @test shape(mesh) == before
        @test length(mesh.nodes) == getnnodes(base)
        @test isempty(mesh.midpoints)
        grid = creategrid(mesh)
        @test all(n -> get_node_coordinate(grid, n) == get_node_coordinate(base, n), 1:getnnodes(base))
        for name in ("left", "right", "top", "bottom")
            @test getfacetset(grid, name) == getfacetset(creategrid(BisectionMesh(base)), name)
        end
        # nothing left to coarsen
        coarsen!(mesh, 1:getncells(mesh))
        @test getncells(mesh) == n0
    end

    @testset "coarsening keeps the mesh conforming and consistent" begin
        mesh = BisectionMesh(base)
        for _ in 1:6
            refine!(mesh, rand(rng, 1:getncells(mesh), 8))
        end
        for step in 1:30
            if isodd(step)
                coarsen!(mesh, unique(rand(rng, 1:getncells(mesh), getncells(mesh) ÷ 2)))
            else
                refine!(mesh, rand(rng, 1:getncells(mesh), 3))
            end
            grid = creategrid(mesh)
            @test is_conforming(grid)
            @test all(c -> signed_area(grid, c) > 0, 1:getncells(grid))
            @test sum(c -> signed_area(grid, c), 1:getncells(grid)) ≈ 4.0
            @test sum(name -> facetset_length(grid, getfacetset(grid, name)), ("left", "right", "top", "bottom")) ≈ 8.0
            # every node is used, and every split edge's midpoint still exists
            @test Set(n for c in getcells(grid) for n in c.nodes) == Set(1:getnnodes(grid))
            @test all(((e, m),) -> m <= getnnodes(grid), mesh.midpoints)
            # the edge-to-leaf map matches the leaves
            edges = Dict{Tuple{Int, Int}, Vector{Int}}()
            for (c, t) in enumerate(mesh.tris), e in ((t[1], t[2]), (t[2], t[3]), (t[3], t[1]))
                push!(get!(edges, minmax(e...), Int[]), c)
            end
            @test Dict(e => sort(c) for (e, c) in mesh.edgecells) == edges
        end
        # coarsening everything repeatedly returns to the initial mesh
        for _ in 1:40
            coarsen!(mesh, 1:getncells(mesh))
        end
        @test shape(mesh) == shape(BisectionMesh(base))
    end

    @testset "coarsening needs the whole star, unless require_all_siblings = false" begin
        mesh = BisectionMesh(base)
        refine!(mesh, 5)                           # interior refinement edge: star of 4 cells
        m = length(mesh.nodes)
        star = [c for (c, t) in enumerate(mesh.tris) if m in t]
        @test length(star) == 4
        n = getncells(mesh)
        coarsen!(mesh, star[1:3])
        @test getncells(mesh) == n
        coarsen!(mesh, star[1:1]; require_all_siblings = false)
        @test getncells(mesh) == n - 2
        @test length(mesh.nodes) == m - 1
        # cell sets survive a refine–coarsen cycle
        cellset_grid = generate_grid(Triangle, (2, 2))
        addcellset!(cellset_grid, "left half", x -> x[1] <= 0)
        mesh = BisectionMesh(cellset_grid)
        refine!(mesh, 1:getncells(mesh))
        coarsen!(mesh, 1:getncells(mesh))
        @test getcellset(creategrid(mesh), "left half") == getcellset(cellset_grid, "left half")
    end

    @testset "refine_and_coarsen!" begin
        mesh = BisectionMesh(base)
        refine!(mesh, 1:getncells(mesh))
        grid = creategrid(mesh)
        # coarsen the cells on the left, refine one on the right, against one numbering
        left = [c for c in 1:getncells(grid) if sum(n -> get_node_coordinate(grid, n)[1], getcells(grid, c).nodes) < 0]
        right = [c for c in 1:getncells(grid) if sum(n -> get_node_coordinate(grid, n)[1], getcells(grid, c).nodes) > 0]
        x_right = sum(n -> get_node_coordinate(grid, n), getcells(grid, right[1]).nodes) / 3
        refine_and_coarsen!(mesh, left, right[1:1])
        grid = creategrid(mesh)
        @test is_conforming(grid)
        @test sum(c -> signed_area(grid, c), 1:getncells(grid)) ≈ 4.0
        # the refined cell: its centroid now lies in a cell of a quarter of its area or less
        @test any(1:getncells(grid)) do c
            a, b, d = (get_node_coordinate(grid, n) for n in getcells(grid, c).nodes)
            inside = all(>=(-1.0e-12), (Ferrite.AMR._signed_area(a, b, x_right), Ferrite.AMR._signed_area(b, d, x_right), Ferrite.AMR._signed_area(d, a, x_right)))
            inside && signed_area(grid, c) <= 4 / 64 / 2 + 1.0e-12
        end
        @test count(c -> mesh.levels[c] == 0, 1:getncells(mesh)) > 0      # left half coarsened
        @test_throws ArgumentError refine_and_coarsen!(mesh, [1], [1])
    end

    @testset "input checks" begin
        @test_throws ArgumentError BisectionMesh(generate_grid(Quadrilateral, (2, 2)))
        @test_throws DomainError BisectionMesh(base, -1)
        @test_throws BoundsError refine!(BisectionMesh(base), [33])
        @test_throws BoundsError coarsen!(BisectionMesh(base), [33])
    end
end
