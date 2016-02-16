#version 430

layout(points) in;
layout(triangle_strip, max_vertices = 3) out;

uniform mat4 projection;
uniform mat4 cameraView;
uniform float volumeExtent;
uniform int resolution;

void main()
{
    vec3 voxelCoord = gl_in[0].gl_Position.xyz;

    /*
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
            // Output the reached level as color
            //float level = float(j) / maxLevel;
            //outputColor.x = level;
            //outputColor.y = level;
            //outputColor.z = level;
            //finished = true;

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

                // Accumulate color
                outputColor = texture(brickPool, brickCoords/(volumeRes) + (1.0/volumeRes)/2.0);
                //outputColor.rgb += (1.0 - outputColor.a) * src.rgb * src.a;
                //outputColor.a += (1.0 - outputColor.a) * src.a;

                // More or less: if you hit something, exit
                if(outputColor.a >= 0.001)
                {
                    //outputColor = src;
                    finished = true;
                }
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
    */

    // Now calculate world position
    vec3 worldPos = volumeExtent * (voxelCoord / float(resolution)) - volumeExtent/2;

    // Size
    vec2 halfSize = vec2(0.1, 0.1);

    // Matrix
    mat4 matrix = projection * cameraView;

    // Emit quad
    gl_Position = matrix * vec4(worldPos + vec3(-halfSize.x, -halfSize.y, 0), 1);
    EmitVertex();

    gl_Position = matrix * vec4(worldPos + vec3(halfSize.x, -halfSize.y, 0), 1);
    EmitVertex();

    gl_Position = matrix * vec4(worldPos + vec3(0, halfSize.y, 0), 1);
    EmitVertex();

    EndPrimitive();

}
