# Voxel Subdivision System

The voxel grid in the world is arbitrarily divisable by cubic partitioning, in which a given
voxel of any size can be divided into eight sub-voxels filling the same volume, and each sub-voxel
can be further arbitrarily divided into sub-voxels, down to a level of detail sufficient to
represent the object withing some configurable resolution.

The system should make it possible to be able to "zoom" out to planet-sized objects with retained
resolution suitable for a LOD system, and to be able to "zoom" in to small scales, to be able to
represent sub-millimeter-sized objects too.

The system should be able to use recursive subdivision (octrees).

The system should also be able to be used in conjunction with the physics system for sculpting, i.e.
deformations, breakup into smaller pieces, and contraction and expansion.

The surfaces should also be able use sparse voxel rasterization to subdivide only necessary regions, to
retain computational effiency, accuracy, and storage size.

The system should use adaptive refinement, only subdividing regions with high gradients or detail, to
save memory.

Use interpolation with 3D subdivision schemes to generate smoother, high-resolution volumetric data from
lower-resolution scans.

It should use recursive partitioning to iterate and subdivide voxels.

The voxel subdivision system should be able to work with solids, liquids, and gases, to be able to use
particle accumulation such as snowfall building a cover of snow, evaporation for liquids to gases in
suitable temperature and humidity conditions.
