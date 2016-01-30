#version 430

// In / out
in vec3 fragPos;
layout(location = 0) out vec4 fragColor;

// Uniforms
layout(r32ui, location = 0) uniform readonly uimageBuffer octree;
layout(rgba32f, location = 1) uniform readonly image2D worldPos;
layout(location = 2) uniform sampler3D brickPool;
uniform vec3 camPos;
uniform float stepSize;
uniform vec3 volumeCenter;
uniform float volumeExtent;

// Defines
int maxSteps = 360;
int maxLevel = 8;

// Helper
uint getBit(uint value, uint position)
{
    return (value >> (position-1)) & 1u;
}

uvec3 decodeBrickCoords(uint coded)
{
    uvec3 coords;
    coords.z =  coded & uint(0x000003FF);
    coords.y = (coded & uint(0x000FFC00)) >> 10U;
    coords.x = (coded & uint(0x3FF00000)) >> 20U;
    return coords;
}

vec3 getVolumePos(vec3 worldPos)
{
    return ((worldPos - volumeCenter) / volumeExtent) + 0.5;
}

// Main
void main()
{
    vec3 fragWorldPosition = imageLoad(worldPos, ivec2(gl_FragCoord.xy)).xyz;
    vec3 position;
    //vec3 position = getVolumePos(fragWorldPosition);
    vec3 dir = normalize(fragWorldPosition - camPos);

    // Catch octree content at fragment position
    uint nodeOffset = 0;
    uint childPointer = 0;

    vec3 rayPosition = camPos;
    vec4 outputColor = vec4(0,0,0,1);

    uint nodeTile;
    uint maxDivide;

    // Get first child pointer
    nodeTile = imageLoad(octree, int(0)).x;
    uint firstChildPointer = nodeTile & uint(0x3fffffff);
    childPointer = firstChildPointer;

    bool finished = false;

    for(int i = 0; i < maxSteps; i++)
    {

        rayPosition = rayPosition + stepSize * dir;
        position = getVolumePos(rayPosition);
        childPointer = firstChildPointer;

        for(int j = 1; j <= maxLevel; j++)
        {
            // Determine, in which octant the searched position is
            uvec3 nextOctant = uvec3(0, 0, 0);
            nextOctant.x = uint(2 * position.x);
            nextOctant.y = uint(2 * position.y);
            nextOctant.z = uint(2 * position.z);

            // Make the octant position 1D for the linear memory
            nodeOffset = 2 * (nextOctant.x + 2 * nextOctant.y + 4 * nextOctant.z);

            // The maxdivide bit indicates whether the node has children:
            // 1 means has children
            // 0 means does not have children
            nodeTile = imageLoad(octree, int(nodeOffset + childPointer * 16U)).x;
            uint maxDivide = getBit(nodeTile, 32);

            if(maxDivide == 0)
            {
                // Output the reached level as color
                //float level = float(j) / maxLevel;
                //outputColor.x = level;
                //outputColor.y = level;
                //outputColor.z = level;
                //finished = true;

                uint nodeValue = imageLoad(octree, int(nodeOffset + childPointer *16U) + 1).x;
                uvec3 brickCoords = decodeBrickCoords(nodeValue);
                //outputColor = texture(brickPool, brickCoords);
                //outputColor = vec4(getBit(nodeValue, 32), 0, 0, 1);
                outputColor = vec4(brickCoords/255,1);
                if(getBit(nodeValue, 32) == 1)
                    finished = true;
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

        }

        if(finished)
            break;
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
