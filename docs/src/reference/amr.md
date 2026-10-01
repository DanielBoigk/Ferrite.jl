```@meta
DocTestSetup = :(using Ferrite)
```

# Adaptive mesh refinement

!!! warning "Experimental feature"
    Adaptive mesh refinement is a new and experimental feature. The API documented here may
    change, gain capabilities, or be restructured in minor releases without following semantic
    versioning. Feedback on it is very welcome.

The adaptive mesh refinement (AMR) functionality is built on a forest of octrees, following
the algorithms of `p4est`. For a conceptual introduction see the [AMR topic
guide](../topics/amr.md); for the internals see the [AMR developer
documentation](../devdocs/AMR.md).

## Forest

```@docs
ForestBWG
```

## Refinement and coarsening

```@docs
refine!
refine_all!
coarsen!
refine_and_coarsen!
balanceforest!
```

## Materialization

```@docs
creategrid
```

## Newest vertex bisection

Conforming local refinement of triangle meshes, without hanging nodes. Refinement and
materialization use the [`refine!`](@ref Ferrite.AMR.refine!) and
[`creategrid`](@ref Ferrite.AMR.creategrid) methods documented above.

```@docs
BisectionMesh
```

## Constraints

```@docs
ConformityConstraint
```
