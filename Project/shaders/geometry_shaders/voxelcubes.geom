#version 430

layout(points) in;
layout(triangle_strip, max_vertices = 24) out;
flat out vec4 col;

layout(binding = 0, r32ui) uniform readonly uimageBuffer octree;
uniform sampler3D brickPool;

uniform mat4 projection;
uniform mat4 cameraView;
uniform float volumeExtent;
uniform int resolution;
uniform int maxLevel;

// Defines
const uint pow2[] = {1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024};
const uvec3 insertPositions[] = {
    uvec3(0, 0, 0),
    uvec3(1, 0, 0),
    uvec3(0, 1, 0),
    uvec3(1, 1, 0),
    uvec3(0, 0, 1),
    uvec3(1, 0, 1),
    uvec3(0, 1, 1),
    uvec3(1, 1, 1)};

// Helper
uint getBit(uint value, uint position)
{
    return (value >> (position-1)) & 1u;
}

uvec3 decodeBrickCoords(uint coded)
{
    uvec3 coords;
    coords.z =  coded & 0x000003FF;
    coords.y = (coded & 0x000FFC00) >> 10U;
    coords.x = (coded & 0x3FF00000) >> 20U;
    return coords;
}

void main()
{
    // Relative position of voxel
    vec3 relativePos = gl_in[0].gl_Position.xyz;

    // Octree reading preparation
    uint nodeOffset = 0;
    uint childPointer = 0;
    uint nodeTile;

    // Some variable renaming to hold name scheme of raycasting shader
    vec3 innerOctreePosition = relativePos;
    float volumeRes = resolution;

    // Get first child pointer
    nodeTile = imageLoad(octree, int(0)).x;
    childPointer = nodeTile & uint(0x3fffffff);

    // Color of voxel
    col = vec4(0,0,0,0);

    // Determine content of octree on current position
    for(int j = 1; j < maxLevel; j++)
    {
        // Determine, in which octant the searched position is
        uvec3 nextOctant = uvec3(0, 0, 0);
        nextOctant.x = uint(2 * innerOctreePosition.x);
        nextOctant.y = uint(2 * innerOctreePosition.y);
        nextOctant.z = uint(2 * innerOctreePosition.z);

        // Make the octant position 1D for the linear memory
        nodeOffset = 2 * (nextOctant.x + 2 * nextOctant.y + 4 * nextOctant.z);
        nodeTile = imageLoad(octree, int(childPointer * 16U + nodeOffset)).x;

        // Update position in volume
        innerOctreePosition.x = 2 * innerOctreePosition.x - nextOctant.x;
        innerOctreePosition.y = 2 * innerOctreePosition.y - nextOctant.y;
        innerOctreePosition.z = 2 * innerOctreePosition.z - nextOctant.z;

        // The 32nd bit indicates whether the node has children:
        // 1 means has children
        // 0 means does not have children
        // Only read from brick, if we are at aimed level in octree
        if(j == maxLevel-1)
        {
            // Brick coordinates
            uint brickTile = imageLoad(octree, int(nodeOffset + childPointer *16U)+1).x;
            uvec3 brickCoords = decodeBrickCoords(brickTile);

            // Just a check, whether brick is there
            if(getBit(brickTile, 31) == 1)
            {
                // Here we should intersect our brick seperately
                // Go one octant deeper in this inner loop cicle to determine exact brick coordinate
                nextOctant.x = uint(2 * innerOctreePosition.x);
                nextOctant.y = uint(2 * innerOctreePosition.y);
                nextOctant.z = uint(2 * innerOctreePosition.z);
                uint offset = nextOctant.x + 2 * nextOctant.y + 4 * nextOctant.z;
                brickCoords += insertPositions[offset]*2;

                // Get color from brick
                col = texture(brickPool, brickCoords/(volumeRes) + (1.0/volumeRes)/2.0);
            }

            // Break inner loop
            break;
        }
        else
        {
            // If the node has children we read the pointer to the next nodetile
            childPointer = nodeTile & uint(0x3fffffff);
        }
    }

    // Only continue, if something was found
    if(col.a > 0.25)
    {
        // Now calculate world position
        vec4 pos = vec4(volumeExtent * relativePos - volumeExtent/2, 1);

        // Size
        float size = volumeExtent / float(resolution);
        vec3 offset = vec3(size / 2.0, size / 2.0, size / 2.0);

        // Matrix
        mat4 M = projection * cameraView;

        vec4 A = vec4(-offset.x, offset.y, offset.z, 0);
        vec4 B = vec4(-offset.x, -offset.y, offset.z, 0);
        vec4 C = vec4( offset.x, -offset.y, offset.z, 0);
        vec4 D = vec4( offset.x, offset.y, offset.z, 0);
        vec4 E = vec4( offset.x, offset.y, -offset.z, 0);
        vec4 F = vec4( offset.x, -offset.y, -offset.z, 0);
        vec4 G = vec4(-offset.x, -offset.y, -offset.z, 0);
        vec4 H = vec4(-offset.x, offset.y, -offset.z, 0);


        gl_Position = M * ( pos + A); EmitVertex();
        gl_Position = M * ( pos + B); EmitVertex();
        gl_Position = M * ( pos + D); EmitVertex();
        gl_Position = M * ( pos + C); EmitVertex();
        EndPrimitive();

        gl_Position = M * ( pos + H); EmitVertex();
        gl_Position = M * ( pos + G); EmitVertex();
        gl_Position = M * ( pos + A); EmitVertex();
        gl_Position = M * ( pos + B); EmitVertex();
        EndPrimitive();

        gl_Position = M * ( pos + D); EmitVertex();
        gl_Position = M * ( pos + C); EmitVertex();
        gl_Position = M * ( pos + E); EmitVertex();
        gl_Position = M * ( pos + F); EmitVertex();
        EndPrimitive();

        gl_Position = M * ( pos + H); EmitVertex();
        gl_Position = M * ( pos + A); EmitVertex();
        gl_Position = M * ( pos + E); EmitVertex();
        gl_Position = M * ( pos + D); EmitVertex();
        EndPrimitive();

        gl_Position = M * ( pos + B); EmitVertex();
        gl_Position = M * ( pos + G); EmitVertex();
        gl_Position = M * ( pos + C); EmitVertex();
        gl_Position = M * ( pos + F); EmitVertex();
        EndPrimitive();

        gl_Position = M * ( pos + E); EmitVertex();
        gl_Position = M * ( pos + F); EmitVertex();
        gl_Position = M * ( pos + H); EmitVertex();
        gl_Position = M * ( pos + G); EmitVertex();
        EndPrimitive();
    }
}
