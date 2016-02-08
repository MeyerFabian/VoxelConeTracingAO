#ifndef OCTREE_MIPMAPPING_CUH
#define OCTREE_MIPMAPPING_CUH


#include <src/SparseOctree/NodePool.h>
#include "brickUtilities.cuh"
#include "traverseKernels.cuh"
#include "bitUtilities.cuh"

__device__
uchar4 avgColor(const uchar4 &c1, const uchar4 &c2)
{
    if(c1.w == 0 && c2.w > 0)
        return c2;

    if(c1.w > 0 && c2.w == 0)
        return c1;

    return make_uchar4((c1.x+c2.x)/2,(c1.y+c2.y)/2,(c1.z+c2.z)/2,(c1.w+c2.w)/2);
}

__device__
void mipMapIsotropic(const uint3 &targetBrick, const uint3 *sourceBricks)
{
    struct child
    {
        uchar4 childColors[27];
    } myChilds[8]; // 8 children with 27 color-values each

    uint3 coords, tmpCoords;
    //load colors YAY pragma unroll works here :D
    #pragma unroll 8
    for(int i=0;i<8;i++)
    {
        coords = sourceBricks[i];
        for(int j=0;j<3;j++)
        {
            //TODO: lookup table
            tmpCoords.x = coords.x + constLookUp1Dto3DIndex[j].x;//coords.x + j / 9;
            tmpCoords.y = coords.y + constLookUp1Dto3DIndex[j].y;//(j / 3) % 3;
            tmpCoords.z = coords.z + constLookUp1Dto3DIndex[j].z;//j % 3;

            surf3Dread(&myChilds[i].childColors[j], colorBrickPool, (tmpCoords.x) * sizeof(uchar4), tmpCoords.y, tmpCoords.z);

            tmpCoords.x = coords.x + constLookUp1Dto3DIndex[j+3].x;//coords.x + j / 9;
            tmpCoords.y = coords.y + constLookUp1Dto3DIndex[j+3].y;//(j / 3) % 3;
            tmpCoords.z = coords.z + constLookUp1Dto3DIndex[j+3].z;//j % 3;

            surf3Dread(&myChilds[i].childColors[j+3], colorBrickPool, (tmpCoords.x) * sizeof(uchar4), tmpCoords.y, tmpCoords.z);

            tmpCoords.x = coords.x + constLookUp1Dto3DIndex[j+6].x;//coords.x + j / 9;
            tmpCoords.y = coords.y + constLookUp1Dto3DIndex[j+6].y;//(j / 3) % 3;
            tmpCoords.z = coords.z + constLookUp1Dto3DIndex[j+6].z;//j % 3;

            surf3Dread(&myChilds[i].childColors[j+6], colorBrickPool, (tmpCoords.x) * sizeof(uchar4), tmpCoords.y, tmpCoords.z);

            tmpCoords.x = coords.x + constLookUp1Dto3DIndex[j+9].x;//coords.x + j / 9;
            tmpCoords.y = coords.y + constLookUp1Dto3DIndex[j+9].y;//(j / 3) % 3;
            tmpCoords.z = coords.z + constLookUp1Dto3DIndex[j+9].z;//j % 3;

            surf3Dread(&myChilds[i].childColors[j+9], colorBrickPool, (tmpCoords.x) * sizeof(uchar4), tmpCoords.y, tmpCoords.z);

            tmpCoords.x = coords.x + constLookUp1Dto3DIndex[j+12].x;//coords.x + j / 9;
            tmpCoords.y = coords.y + constLookUp1Dto3DIndex[j+12].y;//(j / 3) % 3;
            tmpCoords.z = coords.z + constLookUp1Dto3DIndex[j+12].z;//j % 3;

            surf3Dread(&myChilds[i].childColors[j+12], colorBrickPool, (tmpCoords.x) * sizeof(uchar4), tmpCoords.y, tmpCoords.z);

            tmpCoords.x = coords.x + constLookUp1Dto3DIndex[j+15].x;//coords.x + j / 9;
            tmpCoords.y = coords.y + constLookUp1Dto3DIndex[j+15].y;//(j / 3) % 3;
            tmpCoords.z = coords.z + constLookUp1Dto3DIndex[j+15].z;//j % 3;

            surf3Dread(&myChilds[i].childColors[j+15], colorBrickPool, (tmpCoords.x) * sizeof(uchar4), tmpCoords.y, tmpCoords.z);

            tmpCoords.x = coords.x + constLookUp1Dto3DIndex[j+18].x;//coords.x + j / 9;
            tmpCoords.y = coords.y + constLookUp1Dto3DIndex[j+18].y;//(j / 3) % 3;
            tmpCoords.z = coords.z + constLookUp1Dto3DIndex[j+18].z;//j % 3;

            surf3Dread(&myChilds[i].childColors[j+18], colorBrickPool, (tmpCoords.x) * sizeof(uchar4), tmpCoords.y, tmpCoords.z);

            tmpCoords.x = coords.x + constLookUp1Dto3DIndex[j+21].x;//coords.x + j / 9;
            tmpCoords.y = coords.y + constLookUp1Dto3DIndex[j+21].y;//(j / 3) % 3;
            tmpCoords.z = coords.z + constLookUp1Dto3DIndex[j+21].z;//j % 3;

            surf3Dread(&myChilds[i].childColors[j+21], colorBrickPool, (tmpCoords.x) * sizeof(uchar4), tmpCoords.y, tmpCoords.z);

            tmpCoords.x = coords.x + constLookUp1Dto3DIndex[j+24].x;//coords.x + j / 9;
            tmpCoords.y = coords.y + constLookUp1Dto3DIndex[j+24].y;//(j / 3) % 3;
            tmpCoords.z = coords.z + constLookUp1Dto3DIndex[j+24].z;//j % 3;

            surf3Dread(&myChilds[i].childColors[j+24], colorBrickPool, (tmpCoords.x) * sizeof(uchar4), tmpCoords.y, tmpCoords.z);
        }
    }

    // TODO: mipmap

    surf3Dwrite(make_uchar4(0,0,0,0), colorBrickPool, (targetBrick.x) * sizeof(uchar4), 0 + targetBrick.y,
                0 + targetBrick.z);
}

__global__
void combineBrickBorders(node *nodePool,
                         neighbours* neighbourPool,
                         uint1* positionBuffer,
                         unsigned int level,
                         unsigned int fragmentListSize)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if(index >= fragmentListSize)
        return;

    float3 position;
    getVoxelPositionUINTtoFLOAT3(positionBuffer[index].x,position);

    unsigned int foundOn = 0;
    unsigned int childPointer = 0;
    unsigned int offset=0;
    unsigned int nodeTile = 0;
    unsigned int value = 0;
    bool found = false;

    if(level != 0)
        level--;

    //TODO: start threads for every possible node on a level. should be faster!
    offset = traverseToCorrespondingNode(nodePool,position,foundOn,level);

    if(foundOn == level)
        found = true;

    // we found a valid node on the level
    if(found)
    {
        neighbours n = neighbourPool[offset];

        // here we have our brick
        uint3 brickCoords = decodeBrickCoords(nodePool[offset].value);

        uchar4 myColors[9];
        uchar4 neighbourColors[9];

        for(int i=0;i<9;i++)
        {
            myColors[i] = make_uchar4(0, 0, 0, 0);
            neighbourColors[i] = make_uchar4(0, 0, 0, 0);
        }
        

        // load all 6 neighbours
        uint3 nXbrickCoords = decodeBrickCoords(nodePool[n.X].value);
        uint3 nYbrickCoords = decodeBrickCoords(nodePool[n.Y].value);
        uint3 nZbrickCoords = decodeBrickCoords(nodePool[n.Z].value);
        uint3 nNegXbrickCoords = decodeBrickCoords(nodePool[n.negX].value);
        uint3 nNegYbrickCoords = decodeBrickCoords(nodePool[n.negY].value);
        uint3 nNegZbrickCoords = decodeBrickCoords(nodePool[n.negZ].value);

        if(n.Y != 0) {
            // TOP
            surf3Dread(&myColors[0], colorBrickPool, (0 + brickCoords.x) * sizeof(uchar4), 0 + brickCoords.y,
                       0 + brickCoords.z);
            surf3Dread(&myColors[1], colorBrickPool, (1 + brickCoords.x) * sizeof(uchar4), 0 + brickCoords.y,
                       0 + brickCoords.z);
            surf3Dread(&myColors[2], colorBrickPool, (2 + brickCoords.x) * sizeof(uchar4), 0 + brickCoords.y,
                       0 + brickCoords.z);
            surf3Dread(&myColors[3], colorBrickPool, (0 + brickCoords.x) * sizeof(uchar4), 0 + brickCoords.y,
                       1 + brickCoords.z);
            surf3Dread(&myColors[4], colorBrickPool, (1 + brickCoords.x) * sizeof(uchar4), 0 + brickCoords.y,
                       1 + brickCoords.z);
            surf3Dread(&myColors[5], colorBrickPool, (2 + brickCoords.x) * sizeof(uchar4), 0 + brickCoords.y,
                       1 + brickCoords.z);
            surf3Dread(&myColors[6], colorBrickPool, (0 + brickCoords.x) * sizeof(uchar4), 0 + brickCoords.y,
                       2 + brickCoords.z);
            surf3Dread(&myColors[7], colorBrickPool, (1 + brickCoords.x) * sizeof(uchar4), 0 + brickCoords.y,
                       2 + brickCoords.z);
            surf3Dread(&myColors[8], colorBrickPool, (2 + brickCoords.x) * sizeof(uchar4), 0 + brickCoords.y,
                       2 + brickCoords.z);

            surf3Dread(&neighbourColors[0], colorBrickPool, (0 + nYbrickCoords.x) * sizeof(uchar4),
                       2 + nYbrickCoords.y, 0 + nYbrickCoords.z);
            surf3Dread(&neighbourColors[1], colorBrickPool, (1 + nYbrickCoords.x) * sizeof(uchar4),
                       2 + nYbrickCoords.y, 0 + nYbrickCoords.z);
            surf3Dread(&neighbourColors[2], colorBrickPool, (2 + nYbrickCoords.x) * sizeof(uchar4),
                       2 + nYbrickCoords.y, 0 + nYbrickCoords.z);
            surf3Dread(&neighbourColors[3], colorBrickPool, (0 + nYbrickCoords.x) * sizeof(uchar4),
                       2 + nYbrickCoords.y, 1 + nYbrickCoords.z);
            surf3Dread(&neighbourColors[4], colorBrickPool, (1 + nYbrickCoords.x) * sizeof(uchar4),
                       2 + nYbrickCoords.y, 1 + nYbrickCoords.z);
            surf3Dread(&neighbourColors[5], colorBrickPool, (2 + nYbrickCoords.x) * sizeof(uchar4),
                       2 + nYbrickCoords.y, 1 + nYbrickCoords.z);
            surf3Dread(&neighbourColors[6], colorBrickPool, (0 + nYbrickCoords.x) * sizeof(uchar4),
                       2 + nYbrickCoords.y, 2 + nYbrickCoords.z);
            surf3Dread(&neighbourColors[7], colorBrickPool, (1 + nYbrickCoords.x) * sizeof(uchar4),
                       2 + nYbrickCoords.y, 2 + nYbrickCoords.z);
            surf3Dread(&neighbourColors[8], colorBrickPool, (2 + nYbrickCoords.x) * sizeof(uchar4),
                       2 + nYbrickCoords.y, 2 + nYbrickCoords.z);




            /*
            printf("neighbourColor: %d, %d, %d, NY:%d \n", neighbourColors[0].x, neighbourColors[0].y,
                   neighbourColors[0].z, neighbourColors[0].w);
            printf("myColor: %d, %d, %d, NY:%d \n", myColors[0].x, myColors[0].y,
                   myColors[0].z, myColors[0].w);
*/
            __syncthreads();
            uchar4 tmp = avgColor(myColors[0], neighbourColors[0]);
            surf3Dwrite(tmp, colorBrickPool, (0 + brickCoords.x) * sizeof(uchar4), 0 + brickCoords.y,
                        0 + brickCoords.z);
            surf3Dwrite(tmp, colorBrickPool, (0 + nYbrickCoords.x) * sizeof(uchar4), 2 + nYbrickCoords.y,
                        0 + nYbrickCoords.z);

            tmp = avgColor(myColors[1], neighbourColors[1]);
            surf3Dwrite(tmp, colorBrickPool, (1 + brickCoords.x) * sizeof(uchar4), 0 + brickCoords.y,
                        0 + brickCoords.z);
            surf3Dwrite(tmp, colorBrickPool, (1 + nYbrickCoords.x) * sizeof(uchar4), 2 + nYbrickCoords.y,
                        0 + nYbrickCoords.z);

            tmp = avgColor(myColors[2], neighbourColors[2]);
            surf3Dwrite(tmp, colorBrickPool, (2 + brickCoords.x) * sizeof(uchar4), 0 + brickCoords.y,
                        0 + brickCoords.z);
            surf3Dwrite(tmp, colorBrickPool, (2 + nYbrickCoords.x) * sizeof(uchar4), 2 + nYbrickCoords.y,
                        0 + nYbrickCoords.z);

            tmp = avgColor(myColors[3], neighbourColors[3]);
            surf3Dwrite(tmp, colorBrickPool, (0 + brickCoords.x) * sizeof(uchar4), 0 + brickCoords.y,
                        1 + brickCoords.z);
            surf3Dwrite(tmp, colorBrickPool, (0 + nYbrickCoords.x) * sizeof(uchar4), 2 + nYbrickCoords.y,
                        1 + nYbrickCoords.z);

            tmp = avgColor(myColors[4], neighbourColors[4]);
            surf3Dwrite(tmp, colorBrickPool, (1 + brickCoords.x) * sizeof(uchar4), 0 + brickCoords.y,
                        0 + brickCoords.z);
            surf3Dwrite(tmp, colorBrickPool, (1 + nYbrickCoords.x) * sizeof(uchar4), 2 + nYbrickCoords.y,
                        1 + nYbrickCoords.z);

            tmp = avgColor(myColors[5], neighbourColors[5]);
            surf3Dwrite(tmp, colorBrickPool, (2 + brickCoords.x) * sizeof(uchar4), 0 + brickCoords.y,
                        0 + brickCoords.z);
            surf3Dwrite(tmp, colorBrickPool, (2 + nYbrickCoords.x) * sizeof(uchar4), 2 + nYbrickCoords.y,
                        1 + nYbrickCoords.z);


            tmp = avgColor(myColors[6], neighbourColors[6]);
            surf3Dwrite(tmp, colorBrickPool, (0 + brickCoords.x) * sizeof(uchar4), 0 + brickCoords.y,
                        0 + brickCoords.z);
            surf3Dwrite(tmp, colorBrickPool, (0 + nYbrickCoords.x) * sizeof(uchar4), 2 + nYbrickCoords.y,
                        0 + nYbrickCoords.z);

            tmp = avgColor(myColors[7], neighbourColors[7]);
            surf3Dwrite(tmp, colorBrickPool, (1 + brickCoords.x) * sizeof(uchar4), 0 + brickCoords.y,
                        2 + brickCoords.z);
            surf3Dwrite(tmp, colorBrickPool, (1 + nYbrickCoords.x) * sizeof(uchar4), 2 + nYbrickCoords.y,
                        2 + nYbrickCoords.z);

            tmp = avgColor(myColors[8], neighbourColors[8]);
            surf3Dwrite(tmp, colorBrickPool, (2 + brickCoords.x) * sizeof(uchar4), 0 + brickCoords.y,
                        2 + brickCoords.z);
            surf3Dwrite(tmp, colorBrickPool, (2 + nYbrickCoords.x) * sizeof(uchar4), 2 + nYbrickCoords.y,
                        2 + nYbrickCoords.z);
        }


        if(n.negY != 0) {
            // Bottom
            surf3Dread(&myColors[0], colorBrickPool, (0 + brickCoords.x) * sizeof(uchar4), 2 + brickCoords.y,
                       0 + brickCoords.z);
            surf3Dread(&myColors[1], colorBrickPool, (1 + brickCoords.x) * sizeof(uchar4), 2 + brickCoords.y,
                       0 + brickCoords.z);
            surf3Dread(&myColors[2], colorBrickPool, (2 + brickCoords.x) * sizeof(uchar4), 2 + brickCoords.y,
                       0 + brickCoords.z);
            surf3Dread(&myColors[3], colorBrickPool, (0 + brickCoords.x) * sizeof(uchar4), 2 + brickCoords.y,
                       1 + brickCoords.z);
            surf3Dread(&myColors[4], colorBrickPool, (1 + brickCoords.x) * sizeof(uchar4), 2 + brickCoords.y,
                       1 + brickCoords.z);
            surf3Dread(&myColors[5], colorBrickPool, (2 + brickCoords.x) * sizeof(uchar4), 2 + brickCoords.y,
                       1 + brickCoords.z);
            surf3Dread(&myColors[6], colorBrickPool, (0 + brickCoords.x) * sizeof(uchar4), 2 + brickCoords.y,
                       2 + brickCoords.z);
            surf3Dread(&myColors[7], colorBrickPool, (1 + brickCoords.x) * sizeof(uchar4), 2 + brickCoords.y,
                       2 + brickCoords.z);
            surf3Dread(&myColors[8], colorBrickPool, (2 + brickCoords.x) * sizeof(uchar4), 2 + brickCoords.y,
                       2 + brickCoords.z);

            surf3Dread(&neighbourColors[0], colorBrickPool, (0 + nNegYbrickCoords.x) * sizeof(uchar4),
                       0 + nNegYbrickCoords.y, 0 + nNegYbrickCoords.z);
            surf3Dread(&neighbourColors[1], colorBrickPool, (1 + nNegYbrickCoords.x) * sizeof(uchar4),
                       0 + nNegYbrickCoords.y, 0 + nNegYbrickCoords.z);
            surf3Dread(&neighbourColors[2], colorBrickPool, (2 + nNegYbrickCoords.x) * sizeof(uchar4),
                       0 + nNegYbrickCoords.y, 0 + nNegYbrickCoords.z);
            surf3Dread(&neighbourColors[3], colorBrickPool, (0 + nNegYbrickCoords.x) * sizeof(uchar4),
                       0 + nNegYbrickCoords.y, 1 + nNegYbrickCoords.z);
            surf3Dread(&neighbourColors[4], colorBrickPool, (1 + nNegYbrickCoords.x) * sizeof(uchar4),
                       0 + nNegYbrickCoords.y, 1 + nNegYbrickCoords.z);
            surf3Dread(&neighbourColors[5], colorBrickPool, (2 + nNegYbrickCoords.x) * sizeof(uchar4),
                       0 + nNegYbrickCoords.y, 1 + nNegYbrickCoords.z);
            surf3Dread(&neighbourColors[6], colorBrickPool, (0 + nNegYbrickCoords.x) * sizeof(uchar4),
                       0 + nNegYbrickCoords.y, 2 + nNegYbrickCoords.z);
            surf3Dread(&neighbourColors[7], colorBrickPool, (1 + nNegYbrickCoords.x) * sizeof(uchar4),
                       0 + nNegYbrickCoords.y, 2 + nNegYbrickCoords.z);
            surf3Dread(&neighbourColors[8], colorBrickPool, (2 + nNegYbrickCoords.x) * sizeof(uchar4),
                       0 + nNegYbrickCoords.y, 2 + nNegYbrickCoords.z);

            __syncthreads();


            /*
            printf("neighbourColor: %d, %d, %d, NY:%d \n", neighbourColors[0].x, neighbourColors[0].y,
                   neighbourColors[0].z, neighbourColors[0].w);
            printf("myColor: %d, %d, %d, NY:%d \n", myColors[0].x, myColors[0].y,
                   myColors[0].z, myColors[0].w);
*/
            if(myColors[0].w != 0 && neighbourColors[0].w != 0)
            {
                uchar4 tmp = avgColor(myColors[0], neighbourColors[0]);
                surf3Dwrite(tmp, colorBrickPool, (0 + brickCoords.x) * sizeof(uchar4), 2 + brickCoords.y,
                            0 + brickCoords.z);
                surf3Dwrite(tmp, colorBrickPool, (0 + nNegYbrickCoords.x) * sizeof(uchar4), 0 + nNegYbrickCoords.y,
                            0 + nNegYbrickCoords.z);
            }

            if(myColors[1].w != 0 && neighbourColors[1].w != 0)
            {
                uchar4 tmp = avgColor(myColors[1], neighbourColors[1]);
                surf3Dwrite(tmp, colorBrickPool, (1 + brickCoords.x) * sizeof(uchar4), 2 + brickCoords.y,
                            0 + brickCoords.z);
                surf3Dwrite(tmp, colorBrickPool, (1 + nNegYbrickCoords.x) * sizeof(uchar4), 0 + nNegYbrickCoords.y,
                            0 + nNegYbrickCoords.z);
            }

            if(myColors[2].w != 0 && neighbourColors[2].w != 0)
            {
                uchar4 tmp = avgColor(myColors[2], neighbourColors[2]);
                surf3Dwrite(tmp, colorBrickPool, (2 + brickCoords.x) * sizeof(uchar4), 2 + brickCoords.y,
                            0 + brickCoords.z);
                surf3Dwrite(tmp, colorBrickPool, (2 + nNegYbrickCoords.x) * sizeof(uchar4), 0 + nNegYbrickCoords.y,
                            0 + nNegYbrickCoords.z);
            }

            if(myColors[3].w != 0 && neighbourColors[3].w != 0)
            {
                uchar4 tmp = avgColor(myColors[3], neighbourColors[3]);
                surf3Dwrite(tmp, colorBrickPool, (0 + brickCoords.x) * sizeof(uchar4), 2 + brickCoords.y,
                            1 + brickCoords.z);
                surf3Dwrite(tmp, colorBrickPool, (0 + nNegYbrickCoords.x) * sizeof(uchar4), 0 + nNegYbrickCoords.y,
                            1 + nNegYbrickCoords.z);
            }

            if(myColors[4].w != 0 && neighbourColors[4].w != 0)
            {
                uchar4 tmp = avgColor(myColors[4], neighbourColors[4]);
                surf3Dwrite(tmp, colorBrickPool, (1 + brickCoords.x) * sizeof(uchar4), 2 + brickCoords.y,
                            1 + brickCoords.z);
                surf3Dwrite(tmp, colorBrickPool, (1 + nNegYbrickCoords.x) * sizeof(uchar4), 0 + nNegYbrickCoords.y,
                            1 + nNegYbrickCoords.z);
            }

            if(myColors[5].w != 0 && neighbourColors[5].w != 0)
            {
                uchar4 tmp = avgColor(myColors[5], neighbourColors[5]);
                surf3Dwrite(tmp, colorBrickPool, (2 + brickCoords.x) * sizeof(uchar4), 2 + brickCoords.y,
                            1 + brickCoords.z);
                surf3Dwrite(tmp, colorBrickPool, (2 + nNegYbrickCoords.x) * sizeof(uchar4), 0 + nNegYbrickCoords.y,
                            1 + nNegYbrickCoords.z);
            }

            if(myColors[6].w != 0 && neighbourColors[6].w != 0)
            {
                uchar4 tmp = avgColor(myColors[6], neighbourColors[6]);
                surf3Dwrite(tmp, colorBrickPool, (0 + brickCoords.x) * sizeof(uchar4), 2 + brickCoords.y,
                            2 + brickCoords.z);
                surf3Dwrite(tmp, colorBrickPool, (0 + nNegYbrickCoords.x) * sizeof(uchar4), 0 + nNegYbrickCoords.y,
                            2 + nNegYbrickCoords.z);
            }

            if(myColors[7].w != 0 && neighbourColors[7].w != 0)
            {
                uchar4 tmp = avgColor(myColors[7], neighbourColors[7]);
                surf3Dwrite(tmp, colorBrickPool, (1 + brickCoords.x) * sizeof(uchar4), 2 + brickCoords.y,
                            2 + brickCoords.z);
                surf3Dwrite(tmp, colorBrickPool, (1 + nNegYbrickCoords.x) * sizeof(uchar4), 0 + nNegYbrickCoords.y,
                            2 + nNegYbrickCoords.z);
            }

            if(myColors[8].w != 0 && neighbourColors[8].w != 0)
            {
                uchar4 tmp = avgColor(myColors[8], neighbourColors[8]);
                surf3Dwrite(tmp, colorBrickPool, (2 + brickCoords.x) * sizeof(uchar4), 2 + brickCoords.y,
                            2 + brickCoords.z);
                surf3Dwrite(tmp, colorBrickPool, (2 + nNegYbrickCoords.x) * sizeof(uchar4), 0 + nNegYbrickCoords.y,
                            2 + nNegYbrickCoords.z);
            }
        }

    }
}

__global__
void mipMapOctreeLevel(node *nodePool, unsigned int level)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    // make sure our index matches the node-adresses in a given octree level
    index += constLevelIntervalMap[level].start;
    // make sure we dont load invalid adresses
    if(index > constLevelIntervalMap[level].end)
        return;

    // load the target node that should be filled by mipmapping
    node targetNode = nodePool[index];

    // get the childpointer to the node tile
    unsigned int childPointer = targetNode.nodeTilePointer & 0x3fffffff;

    // load the texture coordinates of the associated Brick
    uint3 targetBrick = decodeBrickCoords(targetNode.value);

    if(childPointer != 0)
    {
        uint3 brickCoords[8];

        // load all child-bricks todo: unroll
        for(int i=0;i<8;i++)
        {
            // we have 8 associated nodes in a nodetile
            brickCoords[i] = decodeBrickCoords(nodePool[childPointer+i].value);
        }

        // finally mipmap our node
        mipMapIsotropic(targetBrick,brickCoords);
    }
}

#endif