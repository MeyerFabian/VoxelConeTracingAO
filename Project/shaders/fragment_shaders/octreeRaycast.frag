#version 430

// In / out
in vec3 fragPos;
layout(location = 0) out vec4 fragColor;

// Uniforms
layout(r32ui, location = 0) uniform readonly uimageBuffer octree;
layout(rgba32f, location = 1) uniform readonly image2D worldPos;
uniform vec3 camPos;
uniform float stepSize;
uniform vec3 volumeCenter;
uniform float volumeExtent;

// Defines
int maxSteps = 100;
int maxLevel = 8;

// Helper
uint getBit(uint value, uint position)
{
    return (value >> (position-1)) & 1u;
}

// Main
void main()
{
    vec3 fragWorldPosition = imageLoad(worldPos, ivec2(gl_FragCoord.xy)).xyz;
    vec3 fragVolumePosition = clamp(((fragWorldPosition - volumeCenter) / volumeExtent) + 0.5, 0, 1);
    vec3 dir = normalize(fragWorldPosition - camPos);

    // Catch octree content at fragment position
    uint nodeOffset = 0;
    uint childPointer = 0;

    vec3 position = fragVolumePosition;
    vec4 outputColor = vec4(0,0,0,1);

    // Get first child pointer
    uint nodeTile = imageLoad(octree, int(0)).x;
    childPointer = nodeTile & uint(0x3fffffff);

    for(int j = 1; j <= maxLevel; j++)
    {
        // Determine, in which octant the searched position is
        uvec3 nextOctant = uvec3(0, 0, 0);
        nextOctant.x = uint(2 * position.x);
        nextOctant.y = uint(2 * position.y);
        nextOctant.z = uint(2 * position.z);

        // Make the octant position 1D for the linear memory
        nodeOffset = 2 * (nextOctant.x + 2 * nextOctant.y + 4 * nextOctant.z);

        // The maxdivide bit indicates wheather the node has children:
        // 1 means has children
        // 0 means does not have children
        nodeTile = imageLoad(octree, int(nodeOffset + childPointer * 16U)).x;
        uint maxDivide = getBit(nodeTile, 32);

        if(maxDivide == 0)
        {
            // Output the reached level as color
            float level = float(j) / float(maxLevel);
            outputColor.x = level;
            break;
        }
        else
        {
            // If the node has children we read the pointer to the next nodetile
            childPointer = nodeTile & uint(0x3fffffff);
        }

        // Update position
        position.x = 2 * position.x - nextOctant.x;
        position.y = 2 * position.y - nextOctant.y;
        position.z = 2 * position.z - nextOctant.z;

        // JUST SET SOME SHIT IN THE OUTPUT TO TEST IF THAT SHIT EVEN ENTERS THE SHITTY FOR LOOP
        outputColor.y = 1;
    }

    fragColor = outputColor;
}






























  /*
    vec4 voxelColor = vec4(1.0f, 1.0f, 0.0f, 1.0f);
    vec3 curPos = fragPos;

    // TODO: Kamera rotation nicht beachtet
    vec3 dir = fragPos - camPos;
    dir = normalize(dir);

    uint nodeOffset = 0;
    uint childPointer = 0;
    bool finished = false;

    uint nodeTile = imageLoad(octree, int(nodeOffset + childPointer * 16U)).x;

    for(int i = 0; i < maxSteps; i++)
    {
        // propagate curPos along the ray direction
        curPos += (dir * stepSize);
        for(int j = 0; j < maxLevel; j++)
        {
            uvec3 nextOctant = uvec3(0, 0, 0);
            // determine octant for the given voxel
            // TODO: du initialisiert das quad mit -0.5..0.5. Sprich hier kommt teilweise ein negative Oktant
            nextOctant.x = uint(2 * curPos.x);
            nextOctant.y = uint(2 * curPos.y);
            nextOctant.z = uint(2 * curPos.z);

            // make the octant position 1D for the linear memory
            nodeOffset = 2 * (nextOctant.x + 2 * nextOctant.y + 4 * nextOctant.z);

            // the maxdivide bit indicates wheather the node has children:
            // 1 means has children
            // 0 means does not have children
            //uint nodeTile = imageLoad(octree, int(nodeOffset + childPointer * 16U)).x;
            uint maxDivide = getBit(nodeTile, 32);

            //voxelColor = uvec4(nodeTile,nodeTile,nodeTile,1);

            if(maxDivide == 0)
            {
                float greyValue = float(i)/maxSteps;
                //voxelColor = vec4(greyValue, greyValue, greyValue, 1.0f);
                finished = true;
                break;
            }
            else
            {
                // if the node has children we read the pointer to the next nodetile
                childPointer = nodeTile & uint(0x3fffffff);
            }
        }
        if(finished)
            break;
    }

    voxelColor = uvec4(getBit(nodeTile, 31),getBit(nodeTile, 32),getBit(nodeTile, 32),1);
    */
