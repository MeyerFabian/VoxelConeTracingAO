#version 430

in vec3 fragPos;

uniform vec3 camPos;
uniform float stepSize;
uniform usamplerBuffer octree;

layout(location = 0) out vec4 fragColor;

int maxSteps = 100;
int maxLevel = 7;

uint getBit(uint value, uint position)
{
    return (value >> (position-1)) & 1u;
}

void main()
{
    vec4 voxelColor;
    vec3 curPos = fragPos;
    vec3 dir = fragPos - camPos;
    normalize(dir);

    uint nodeOffset = 0;
    uint childPointer = 0;
    bool finished = false;

    for(int i = 0; i < maxSteps; i++)
    {
        curPos += (dir * stepSize);
        for(int j = 0; j < maxLevel; j++)
        {
            uvec3 nextOctant = uvec3(0, 0, 0);
            // determine octant for the given voxel
            nextOctant.x = uint(2 * curPos.x);
            nextOctant.y = uint(2 * curPos.y);
            nextOctant.z = uint(2 * curPos.z);

            // make the octant position 1D for the linear memory
            nodeOffset = 2 * (nextOctant.x + 2 * nextOctant.y + 4 * nextOctant.z);

            // the maxdivide bit indicates wheather the node has children:
            // 1 means has children
            // 0 means does not have children
            uint nodeTile = texelFetch(octree, int(nodeOffset + childPointer * 16U)).r;
            //uint nodeTile = nodePool[nodeOffset+childPointer*8].nodeTilePointer;
            uint maxDivide = getBit(nodeTile, 32);

            if(maxDivide == 0)
            {
                float greyValue = i / maxSteps;
                voxelColor = vec4(greyValue, greyValue, greyValue, 1.0f);
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

    fragColor = voxelColor;
}

