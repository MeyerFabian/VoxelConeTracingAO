#ifndef OCTREE_MIPMAPPING_CUH
#define OCTREE_MIPMAPPING_CUH


#include <src/SparseOctree/NodePool.h>
#include "brickUtilities.cuh"
#include "traverseKernels.cuh"
#include "bitUtilities.cuh"

__device__
uchar4 avgColor(const uchar4 &c1, const uchar4 &c2)
{
    if(c1.w < 255 && c2.w >= 255)
        return c2;

    if(c1.w >= 255 && c2.w < 255)
        return c1;

    return make_uchar4((c1.x+c2.x)/2.0,(c1.y+c2.y)/2.0,(c1.z+c2.z)/2.0,(c1.w+c2.w)/2.0);
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
        for(int j=0;j<27;j++)
        {
            //TODO: lookup table
            tmpCoords.x = coords.x + constLookUp1Dto3DIndex[j].x;//coords.x + j / 9;
            tmpCoords.y = coords.y + constLookUp1Dto3DIndex[j].y;//(j / 3) % 3;
            tmpCoords.z = coords.z + constLookUp1Dto3DIndex[j].z;//j % 3;

            surf3Dread(&myChilds[i].childColors[j], colorBrickPool, (tmpCoords.x) * sizeof(uchar4), tmpCoords.y, tmpCoords.z);
        }
    }

    // TODO: mipmap
    __syncthreads();

    //MIPMAP CENTER:
    float4 centerColor = make_float4(0,0,0,0);
    centerColor.x += 0.25*myChilds[0].childColors[26].x         // center
                     + 0.125*myChilds[0].childColors[17].x      // faces =>
                     + 0.125*myChilds[1].childColors[17].x
                     + 0.125*myChilds[0].childColors[25].x
                     + 0.125*myChilds[4].childColors[25].x
                     + 0.125*myChilds[0].childColors[23].x
                     + 0.125*myChilds[2].childColors[23].x      // <= faces
                     + 0.0625*myChilds[0].childColors[14].x     // => edges
                     + 0.0625*myChilds[2].childColors[14].x
                     + 0.0625*myChilds[0].childColors[16].x
                     + 0.0625*myChilds[4].childColors[16].x
                     + 0.0625*myChilds[1].childColors[14].x
                     + 0.0625*myChilds[3].childColors[14].x
                     + 0.0625*myChilds[1].childColors[16].x
                     + 0.0625*myChilds[5].childColors[16].x
                     + 0.0625*myChilds[0].childColors[22].x
                     + 0.0625*myChilds[2].childColors[22].x
                     + 0.0625*myChilds[4].childColors[22].x
                     + 0.0625*myChilds[6].childColors[22].x    // <= edges
                     + 0.03125*myChilds[0].childColors[13].x   // corners =>
                     + 0.03125*myChilds[2].childColors[13].x
                     + 0.03125*myChilds[1].childColors[13].x
                     + 0.03125*myChilds[3].childColors[13].x
                     + 0.03125*myChilds[4].childColors[13].x
                     + 0.03125*myChilds[6].childColors[13].x
                     + 0.03125*myChilds[5].childColors[13].x
                     + 0.03125*myChilds[7].childColors[13].x; // <= corners

    centerColor.y += 0.25*myChilds[0].childColors[26].y         // center
                     + 0.125*myChilds[0].childColors[17].y      // faces =>
                     + 0.125*myChilds[1].childColors[17].y
                     + 0.125*myChilds[0].childColors[25].y
                     + 0.125*myChilds[4].childColors[25].y
                     + 0.125*myChilds[0].childColors[23].y
                     + 0.125*myChilds[2].childColors[23].y      // <= faces
                     + 0.0625*myChilds[0].childColors[14].y     // => edges
                     + 0.0625*myChilds[2].childColors[14].y
                     + 0.0625*myChilds[0].childColors[16].y
                     + 0.0625*myChilds[4].childColors[16].y
                     + 0.0625*myChilds[1].childColors[14].y
                     + 0.0625*myChilds[3].childColors[14].y
                     + 0.0625*myChilds[1].childColors[16].y
                     + 0.0625*myChilds[5].childColors[16].y
                     + 0.0625*myChilds[0].childColors[22].y
                     + 0.0625*myChilds[2].childColors[22].y
                     + 0.0625*myChilds[4].childColors[22].y
                     + 0.0625*myChilds[6].childColors[22].y    // <= edges
                     + 0.03125*myChilds[0].childColors[13].y   // corners =>
                     + 0.03125*myChilds[2].childColors[13].y
                     + 0.03125*myChilds[1].childColors[13].y
                     + 0.03125*myChilds[3].childColors[13].y
                     + 0.03125*myChilds[4].childColors[13].y
                     + 0.03125*myChilds[6].childColors[13].y
                     + 0.03125*myChilds[5].childColors[13].y
                     + 0.03125*myChilds[7].childColors[13].y; // <= corners

    centerColor.z += 0.25*myChilds[0].childColors[26].z         // center
                     + 0.125*myChilds[0].childColors[17].z      // faces =>
                     + 0.125*myChilds[1].childColors[17].z
                     + 0.125*myChilds[0].childColors[25].z
                     + 0.125*myChilds[4].childColors[25].z
                     + 0.125*myChilds[0].childColors[23].z
                     + 0.125*myChilds[2].childColors[23].z      // <= faces
                     + 0.0625*myChilds[0].childColors[14].z     // => edges
                     + 0.0625*myChilds[2].childColors[14].z
                     + 0.0625*myChilds[0].childColors[16].z
                     + 0.0625*myChilds[4].childColors[16].z
                     + 0.0625*myChilds[1].childColors[14].z
                     + 0.0625*myChilds[3].childColors[14].z
                     + 0.0625*myChilds[1].childColors[16].z
                     + 0.0625*myChilds[5].childColors[16].z
                     + 0.0625*myChilds[0].childColors[22].z
                     + 0.0625*myChilds[2].childColors[22].z
                     + 0.0625*myChilds[4].childColors[22].z
                     + 0.0625*myChilds[6].childColors[22].z    // <= edges
                     + 0.03125*myChilds[0].childColors[13].z   // corners =>
                     + 0.03125*myChilds[2].childColors[13].z
                     + 0.03125*myChilds[1].childColors[13].z
                     + 0.03125*myChilds[3].childColors[13].z
                     + 0.03125*myChilds[4].childColors[13].z
                     + 0.03125*myChilds[6].childColors[13].z
                     + 0.03125*myChilds[5].childColors[13].z
                     + 0.03125*myChilds[7].childColors[13].z; // <= corners

    centerColor.w += 0.25*myChilds[0].childColors[26].w         // center
                     + 0.125*myChilds[0].childColors[17].w      // faces =>
                     + 0.125*myChilds[1].childColors[17].w
                     + 0.125*myChilds[0].childColors[25].w
                     + 0.125*myChilds[4].childColors[25].w
                     + 0.125*myChilds[0].childColors[23].w
                     + 0.125*myChilds[2].childColors[23].w      // <= faces
                     + 0.0625*myChilds[0].childColors[14].w     // => edges
                     + 0.0625*myChilds[2].childColors[14].w
                     + 0.0625*myChilds[0].childColors[16].w
                     + 0.0625*myChilds[4].childColors[16].w
                     + 0.0625*myChilds[1].childColors[14].w
                     + 0.0625*myChilds[3].childColors[14].w
                     + 0.0625*myChilds[1].childColors[16].w
                     + 0.0625*myChilds[5].childColors[16].w
                     + 0.0625*myChilds[0].childColors[22].w
                     + 0.0625*myChilds[2].childColors[22].w
                     + 0.0625*myChilds[4].childColors[22].w
                     + 0.0625*myChilds[6].childColors[22].w    // <= edges
                     + 0.03125*myChilds[0].childColors[13].w   // corners =>
                     + 0.03125*myChilds[2].childColors[13].w
                     + 0.03125*myChilds[1].childColors[13].w
                     + 0.03125*myChilds[3].childColors[13].w
                     + 0.03125*myChilds[4].childColors[13].w
                     + 0.03125*myChilds[6].childColors[13].w
                     + 0.03125*myChilds[5].childColors[13].w
                     + 0.03125*myChilds[7].childColors[13].w; // <= corners

    // TODO: real gaussian kernel for 3D?
    centerColor.x /= 2.0;
    centerColor.y /= 2.0;
    centerColor.z /= 2.0;
    centerColor.w /= 2.0;


    //centerColor = make_float4(127,127,127,255);

    // MIPMAP CORNERS
    float4 leftTopNear = make_float4(0,0,0,0);

    // needed 000, 100, 001, 101, 010, 110, 011, 111

    leftTopNear.x += (0.25*myChilds[0].childColors[0].x
                     + 0.125*myChilds[0].childColors[9].x
                     + 0.125*myChilds[0].childColors[1].x
                     + 0.125*myChilds[0].childColors[3].x
                     + 0.0625*myChilds[0].childColors[10].x
                     + 0.0625*myChilds[0].childColors[12].x
                     + 0.0625*myChilds[0].childColors[4].x
                     + 0.03125*myChilds[0].childColors[13].x);

    leftTopNear.y += (0.25*myChilds[0].childColors[0].y
                     + 0.125*myChilds[0].childColors[9].y
                     + 0.125*myChilds[0].childColors[1].y
                     + 0.125*myChilds[0].childColors[3].y
                     + 0.0625*myChilds[0].childColors[10].y
                     + 0.0625*myChilds[0].childColors[12].y
                     + 0.0625*myChilds[0].childColors[4].y
                     + 0.03125*myChilds[0].childColors[13].y);

    leftTopNear.z += (0.25*myChilds[0].childColors[0].z
                     + 0.125*myChilds[0].childColors[9].z
                     + 0.125*myChilds[0].childColors[1].z
                     + 0.125*myChilds[0].childColors[3].z
                     + 0.0625*myChilds[0].childColors[10].z
                     + 0.0625*myChilds[0].childColors[12].z
                     + 0.0625*myChilds[0].childColors[4].z
                     + 0.03125*myChilds[0].childColors[13].z);

    leftTopNear.w += (0.25*myChilds[0].childColors[0].w
                     + 0.125*myChilds[0].childColors[9].w
                     + 0.125*myChilds[0].childColors[1].w
                     + 0.125*myChilds[0].childColors[3].w
                     + 0.0625*myChilds[0].childColors[10].w
                     + 0.0625*myChilds[0].childColors[12].w
                     + 0.0625*myChilds[0].childColors[4].w
                     + 0.03125*myChilds[0].childColors[13].w);

    leftTopNear.x /= 0.85;
    leftTopNear.y /= 0.85;
    leftTopNear.z /= 0.85;
    leftTopNear.w /= 0.85;

    //leftTopNear = make_float4(255,0,0,255);

    float4 leftBottomNear = make_float4(0,0,0,0);

    // needed 020, 120, 021, 121, 010, 110, 011, 111

    leftBottomNear.x += (0.25*myChilds[2].childColors[6].x
                      + 0.125*myChilds[2].childColors[15].x
                      + 0.125*myChilds[2].childColors[7].x
                      + 0.125*myChilds[2].childColors[3].x
                      + 0.0625*myChilds[2].childColors[16].x
                      + 0.0625*myChilds[2].childColors[12].x
                      + 0.0625*myChilds[2].childColors[4].x
                      + 0.03125*myChilds[2].childColors[13].x);

    leftBottomNear.y += (0.25*myChilds[2].childColors[6].y
                         + 0.125*myChilds[2].childColors[15].y
                         + 0.125*myChilds[2].childColors[7].y
                         + 0.125*myChilds[2].childColors[3].y
                         + 0.0625*myChilds[2].childColors[16].y
                         + 0.0625*myChilds[2].childColors[12].y
                         + 0.0625*myChilds[2].childColors[4].y
                         + 0.03125*myChilds[2].childColors[13].y);

    leftBottomNear.z += (0.25*myChilds[2].childColors[6].z
                         + 0.125*myChilds[2].childColors[15].z
                         + 0.125*myChilds[2].childColors[7].z
                         + 0.125*myChilds[2].childColors[3].z
                         + 0.0625*myChilds[2].childColors[16].z
                         + 0.0625*myChilds[2].childColors[12].z
                         + 0.0625*myChilds[2].childColors[4].z
                         + 0.03125*myChilds[2].childColors[13].z);

    leftBottomNear.w += (0.25*myChilds[2].childColors[6].w
                         + 0.125*myChilds[2].childColors[15].w
                         + 0.125*myChilds[2].childColors[7].w
                         + 0.125*myChilds[2].childColors[3].w
                         + 0.0625*myChilds[2].childColors[16].w
                         + 0.0625*myChilds[2].childColors[12].w
                         + 0.0625*myChilds[2].childColors[4].w
                         + 0.03125*myChilds[2].childColors[13].w);


    leftBottomNear.x /= 0.85;
    leftBottomNear.y /= 0.85;
    leftBottomNear.z /= 0.85;
    leftBottomNear.w /= 0.85;

    //leftBottomNear = make_float4(255,0,255,255);

    float4 leftTopFar = make_float4(0,0,0,0);

    // needed 002, 102, 001, 012, 011, 112, 101, 111
    leftTopFar.x += (0.25*myChilds[4].childColors[2].x
                         + 0.125*myChilds[4].childColors[11].x
                         + 0.125*myChilds[4].childColors[1].x
                         + 0.125*myChilds[4].childColors[5].x
                         + 0.0625*myChilds[4].childColors[4].x
                         + 0.0625*myChilds[4].childColors[14].x
                         + 0.0625*myChilds[4].childColors[10].x
                         + 0.03125*myChilds[4].childColors[13].x);

    leftTopFar.y += (0.25*myChilds[4].childColors[2].y
                     + 0.125*myChilds[4].childColors[11].y
                     + 0.125*myChilds[4].childColors[1].y
                     + 0.125*myChilds[4].childColors[5].y
                     + 0.0625*myChilds[4].childColors[4].y
                     + 0.0625*myChilds[4].childColors[14].y
                     + 0.0625*myChilds[4].childColors[10].y
                     + 0.03125*myChilds[4].childColors[13].y);

    leftTopFar.z += (0.25*myChilds[4].childColors[2].z
                     + 0.125*myChilds[4].childColors[11].z
                     + 0.125*myChilds[4].childColors[1].z
                     + 0.125*myChilds[4].childColors[5].z
                     + 0.0625*myChilds[4].childColors[4].z
                     + 0.0625*myChilds[4].childColors[14].z
                     + 0.0625*myChilds[4].childColors[10].z
                     + 0.03125*myChilds[4].childColors[13].z);

    leftTopFar.w += (0.25*myChilds[4].childColors[2].w
                     + 0.125*myChilds[4].childColors[11].w
                     + 0.125*myChilds[4].childColors[1].w
                     + 0.125*myChilds[4].childColors[5].w
                     + 0.0625*myChilds[4].childColors[4].w
                     + 0.0625*myChilds[4].childColors[14].w
                     + 0.0625*myChilds[4].childColors[10].w
                     + 0.03125*myChilds[4].childColors[13].w);


    leftTopFar.x /= 0.85;
    leftTopFar.y /= 0.85;
    leftTopFar.z /= 0.85;
    leftTopFar.w /= 0.85;

    //leftTopFar = make_float4(255,0,0,255);

    float4 leftBottomFar = make_float4(0,0,0,0);

    // needed 022, 122, 021, 012, 011, 112, 121, 111

    leftBottomFar.x += (0.25*myChilds[6].childColors[8].x
                     + 0.125*myChilds[6].childColors[17].x
                     + 0.125*myChilds[6].childColors[7].x
                     + 0.125*myChilds[6].childColors[5].x
                     + 0.0625*myChilds[6].childColors[4].x
                     + 0.0625*myChilds[6].childColors[14].x
                     + 0.0625*myChilds[6].childColors[16].x
                     + 0.03125*myChilds[6].childColors[13].x);

    leftBottomFar.y += (0.25*myChilds[6].childColors[8].y
                        + 0.125*myChilds[6].childColors[17].y
                        + 0.125*myChilds[6].childColors[7].y
                        + 0.125*myChilds[6].childColors[5].y
                        + 0.0625*myChilds[6].childColors[4].y
                        + 0.0625*myChilds[6].childColors[14].y
                        + 0.0625*myChilds[6].childColors[16].y
                        + 0.03125*myChilds[6].childColors[13].y);

    leftBottomFar.z += (0.25*myChilds[6].childColors[8].z
                        + 0.125*myChilds[6].childColors[17].z
                        + 0.125*myChilds[6].childColors[7].z
                        + 0.125*myChilds[6].childColors[5].z
                        + 0.0625*myChilds[6].childColors[4].z
                        + 0.0625*myChilds[6].childColors[14].z
                        + 0.0625*myChilds[6].childColors[16].z
                        + 0.03125*myChilds[6].childColors[13].z);

    leftBottomFar.w += (0.25*myChilds[6].childColors[8].w
                        + 0.125*myChilds[6].childColors[17].w
                        + 0.125*myChilds[6].childColors[7].w
                        + 0.125*myChilds[6].childColors[5].w
                        + 0.0625*myChilds[6].childColors[4].w
                        + 0.0625*myChilds[6].childColors[14].w
                        + 0.0625*myChilds[6].childColors[16].w
                        + 0.03125*myChilds[6].childColors[13].w);



    leftBottomFar.x /= 0.85;
    leftBottomFar.y /= 0.85;
    leftBottomFar.z /= 0.85;
    leftBottomFar.w /= 0.85;

   // leftBottomFar = make_float4(255,255,0,255);

    float4 rightTopNear = make_float4(0,0,0,0);

    // needed 200, 100, 201, 101,   210, 110, 211, 111

    rightTopNear.x += (0.25*myChilds[1].childColors[18].x
                        + 0.125*myChilds[1].childColors[9].x
                        + 0.125*myChilds[1].childColors[19].x
                        + 0.125*myChilds[1].childColors[21].x
                        + 0.0625*myChilds[1].childColors[10].x
                        + 0.0625*myChilds[1].childColors[12].x
                        + 0.0625*myChilds[1].childColors[22].x
                        + 0.03125*myChilds[1].childColors[13].x);

    rightTopNear.y += (0.25*myChilds[1].childColors[18].y
                       + 0.125*myChilds[1].childColors[9].y
                       + 0.125*myChilds[1].childColors[19].y
                       + 0.125*myChilds[1].childColors[21].y
                       + 0.0625*myChilds[1].childColors[10].y
                       + 0.0625*myChilds[1].childColors[12].y
                       + 0.0625*myChilds[1].childColors[22].y
                       + 0.03125*myChilds[1].childColors[13].y);

    rightTopNear.z += (0.25*myChilds[1].childColors[18].z
                       + 0.125*myChilds[1].childColors[9].z
                       + 0.125*myChilds[1].childColors[19].z
                       + 0.125*myChilds[1].childColors[21].z
                       + 0.0625*myChilds[1].childColors[10].z
                       + 0.0625*myChilds[1].childColors[12].z
                       + 0.0625*myChilds[1].childColors[22].z
                       + 0.03125*myChilds[1].childColors[13].z);

    rightTopNear.w += (0.25*myChilds[1].childColors[18].w
                       + 0.125*myChilds[1].childColors[9].w
                       + 0.125*myChilds[1].childColors[19].w
                       + 0.125*myChilds[1].childColors[21].w
                       + 0.0625*myChilds[1].childColors[10].w
                       + 0.0625*myChilds[1].childColors[12].w
                       + 0.0625*myChilds[1].childColors[22].w
                       + 0.03125*myChilds[1].childColors[13].w);


    rightTopNear.x /= 0.85;
    rightTopNear.y /= 0.85;
    rightTopNear.z /= 0.85;
    rightTopNear.w /= 0.85;

    //rightTopNear = make_float4(0,0,255,255);

    float4 rightBottomNear = make_float4(0,0,0,0);

    // needed 220, 120, 221, 121,   210, 110, 211, 111

    rightBottomNear.x += (0.25*myChilds[3].childColors[24].x
                       + 0.125*myChilds[3].childColors[15].x
                       + 0.125*myChilds[3].childColors[25].x
                       + 0.125*myChilds[3].childColors[21].x
                       + 0.0625*myChilds[3].childColors[16].x
                       + 0.0625*myChilds[3].childColors[12].x
                       + 0.0625*myChilds[3].childColors[22].x
                       + 0.03125*myChilds[3].childColors[13].x);

    rightBottomNear.y += (0.25*myChilds[3].childColors[24].y
                          + 0.125*myChilds[3].childColors[15].y
                          + 0.125*myChilds[3].childColors[25].y
                          + 0.125*myChilds[3].childColors[21].y
                          + 0.0625*myChilds[3].childColors[16].y
                          + 0.0625*myChilds[3].childColors[12].y
                          + 0.0625*myChilds[3].childColors[22].y
                          + 0.03125*myChilds[3].childColors[13].y);

    rightBottomNear.z += (0.25*myChilds[3].childColors[24].z
                          + 0.125*myChilds[3].childColors[15].z
                          + 0.125*myChilds[3].childColors[25].z
                          + 0.125*myChilds[3].childColors[21].z
                          + 0.0625*myChilds[3].childColors[16].z
                          + 0.0625*myChilds[3].childColors[12].z
                          + 0.0625*myChilds[3].childColors[22].z
                          + 0.03125*myChilds[3].childColors[13].z);

    rightBottomNear.w += (0.25*myChilds[3].childColors[24].w
                          + 0.125*myChilds[3].childColors[15].w
                          + 0.125*myChilds[3].childColors[25].w
                          + 0.125*myChilds[3].childColors[21].w
                          + 0.0625*myChilds[3].childColors[16].w
                          + 0.0625*myChilds[3].childColors[12].w
                          + 0.0625*myChilds[3].childColors[22].w
                          + 0.03125*myChilds[3].childColors[13].w);


    rightBottomNear.x /= 0.85;
    rightBottomNear.y /= 0.85;
    rightBottomNear.z /= 0.85;
    rightBottomNear.w /= 0.85;

    //rightBottomNear = make_float4(0,255,0,255);

    float4 rightTopFar = make_float4(0,0,0,0);

    // needed 202, 102, 201, 101, 212, 112, 211 ,111

    rightTopFar.x += (0.25*myChilds[5].childColors[20].x
                          + 0.125*myChilds[5].childColors[11].x
                          + 0.125*myChilds[5].childColors[19].x
                          + 0.125*myChilds[5].childColors[23].x
                          + 0.0625*myChilds[5].childColors[10].x
                          + 0.0625*myChilds[5].childColors[14].x
                          + 0.0625*myChilds[5].childColors[22].x
                          + 0.03125*myChilds[5].childColors[13].x);

    rightTopFar.y += (0.25*myChilds[5].childColors[20].y
                      + 0.125*myChilds[5].childColors[11].y
                      + 0.125*myChilds[5].childColors[19].y
                      + 0.125*myChilds[5].childColors[23].y
                      + 0.0625*myChilds[5].childColors[10].y
                      + 0.0625*myChilds[5].childColors[14].y
                      + 0.0625*myChilds[5].childColors[22].y
                      + 0.03125*myChilds[5].childColors[13].y);

    rightTopFar.z += (0.25*myChilds[5].childColors[20].z
                      + 0.125*myChilds[5].childColors[11].z
                      + 0.125*myChilds[5].childColors[19].z
                      + 0.125*myChilds[5].childColors[23].z
                      + 0.0625*myChilds[5].childColors[10].z
                      + 0.0625*myChilds[5].childColors[14].z
                      + 0.0625*myChilds[5].childColors[22].z
                      + 0.03125*myChilds[5].childColors[13].z);

    rightTopFar.w += (0.25*myChilds[5].childColors[20].w
                      + 0.125*myChilds[5].childColors[11].w
                      + 0.125*myChilds[5].childColors[19].w
                      + 0.125*myChilds[5].childColors[23].w
                      + 0.0625*myChilds[5].childColors[10].w
                      + 0.0625*myChilds[5].childColors[14].w
                      + 0.0625*myChilds[5].childColors[22].w
                      + 0.03125*myChilds[5].childColors[13].w);



    rightTopFar.x /= 0.85;
    rightTopFar.y /= 0.85;
    rightTopFar.z /= 0.85;
    rightTopFar.w /= 0.85;

    //rightTopFar = make_float4(255,0,0,255);

    float4 rightBottomFar = make_float4(0,0,0,0);

    // needed 222, 122, 221, 121, 212, 112, 211 ,111

    rightBottomFar.x += (0.25*myChilds[7].childColors[26].x
                      + 0.125*myChilds[7].childColors[17].x
                      + 0.125*myChilds[7].childColors[25].x
                      + 0.125*myChilds[7].childColors[23].x
                      + 0.0625*myChilds[7].childColors[16].x
                      + 0.0625*myChilds[7].childColors[14].x
                      + 0.0625*myChilds[7].childColors[22].x
                      + 0.03125*myChilds[7].childColors[13].x);

    rightBottomFar.y += (0.25*myChilds[7].childColors[26].y
                         + 0.125*myChilds[7].childColors[17].y
                         + 0.125*myChilds[7].childColors[25].y
                         + 0.125*myChilds[7].childColors[23].y
                         + 0.0625*myChilds[7].childColors[16].y
                         + 0.0625*myChilds[7].childColors[14].y
                         + 0.0625*myChilds[7].childColors[22].y
                         + 0.03125*myChilds[7].childColors[13].y);

    rightBottomFar.z += (0.25*myChilds[7].childColors[26].z
                         + 0.125*myChilds[7].childColors[17].z
                         + 0.125*myChilds[7].childColors[25].z
                         + 0.125*myChilds[7].childColors[23].z
                         + 0.0625*myChilds[7].childColors[16].z
                         + 0.0625*myChilds[7].childColors[14].z
                         + 0.0625*myChilds[7].childColors[22].z
                         + 0.03125*myChilds[7].childColors[13].z);

    rightBottomFar.w += (0.25*myChilds[7].childColors[26].w
                         + 0.125*myChilds[7].childColors[17].w
                         + 0.125*myChilds[7].childColors[25].w
                         + 0.125*myChilds[7].childColors[23].w
                         + 0.0625*myChilds[7].childColors[16].w
                         + 0.0625*myChilds[7].childColors[14].w
                         + 0.0625*myChilds[7].childColors[22].w
                         + 0.03125*myChilds[7].childColors[13].w);



    rightBottomFar.x /= 0.85;
    rightBottomFar.y /= 0.85;
    rightBottomFar.z /= 0.85;
    rightBottomFar.w /= 0.85;

    //mipmap FACES
    // FRONT FACE
    float4 frontFace = make_float4(0,0,0,0);

    // needed in child 0: 110 210 120 220 111 211 121 221
    // needed in child 2: 110 210 111 211
    // needed in child 1: 110 120 111 121
    // needed in child 3: 110 111
    // keep in mind that the backface need less lookups as the faces intersect
    frontFace.x += 0.25*myChilds[0].childColors[24].x
                   + 0.125*myChilds[0].childColors[21].x
                   + 0.125*myChilds[0].childColors[15].x
                   + 0.125*myChilds[0].childColors[25].x
                   + 0.125*myChilds[1].childColors[15].x
                   + 0.125*myChilds[2].childColors[21].x
                   + 0.0625*myChilds[0].childColors[12].x
                   + 0.0625*myChilds[0].childColors[16].x
                   + 0.0625*myChilds[0].childColors[22].x
                   + 0.0625*myChilds[1].childColors[12].x
                   + 0.0625*myChilds[1].childColors[16].x
                   + 0.0625*myChilds[3].childColors[12].x
                   + 0.0625*myChilds[2].childColors[12].x
                   + 0.0625*myChilds[2].childColors[22].x
                   + 0.03125*myChilds[0].childColors[13].x
                   + 0.03125*myChilds[1].childColors[13].x
                   + 0.03125*myChilds[3].childColors[13].x
                   + 0.03125*myChilds[2].childColors[13].x;

    frontFace.y += 0.25*myChilds[0].childColors[24].y
                   + 0.125*myChilds[0].childColors[21].y
                   + 0.125*myChilds[0].childColors[15].y
                   + 0.125*myChilds[0].childColors[25].y
                   + 0.125*myChilds[1].childColors[15].y
                   + 0.125*myChilds[2].childColors[21].y
                   + 0.0625*myChilds[0].childColors[12].y
                   + 0.0625*myChilds[0].childColors[16].y
                   + 0.0625*myChilds[0].childColors[22].y
                   + 0.0625*myChilds[1].childColors[12].y
                   + 0.0625*myChilds[1].childColors[16].y
                   + 0.0625*myChilds[3].childColors[12].y
                   + 0.0625*myChilds[2].childColors[12].y
                   + 0.0625*myChilds[2].childColors[22].y
                   + 0.03125*myChilds[0].childColors[13].y
                   + 0.03125*myChilds[1].childColors[13].y
                   + 0.03125*myChilds[3].childColors[13].y
                   + 0.03125*myChilds[2].childColors[13].y;

    frontFace.z += 0.25*myChilds[0].childColors[24].z
                   + 0.125*myChilds[0].childColors[21].z
                   + 0.125*myChilds[0].childColors[15].z
                   + 0.125*myChilds[0].childColors[25].z
                   + 0.125*myChilds[1].childColors[15].z
                   + 0.125*myChilds[2].childColors[21].z
                   + 0.0625*myChilds[0].childColors[12].z
                   + 0.0625*myChilds[0].childColors[16].z
                   + 0.0625*myChilds[0].childColors[22].z
                   + 0.0625*myChilds[1].childColors[12].z
                   + 0.0625*myChilds[1].childColors[16].z
                   + 0.0625*myChilds[3].childColors[12].z
                   + 0.0625*myChilds[2].childColors[12].z
                   + 0.0625*myChilds[2].childColors[22].z
                   + 0.03125*myChilds[0].childColors[13].z
                   + 0.03125*myChilds[1].childColors[13].z
                   + 0.03125*myChilds[3].childColors[13].z
                   + 0.03125*myChilds[2].childColors[13].z;

    frontFace.w += 0.25*myChilds[0].childColors[24].w
                   + 0.125*myChilds[0].childColors[21].w
                   + 0.125*myChilds[0].childColors[15].w
                   + 0.125*myChilds[0].childColors[25].w
                   + 0.125*myChilds[1].childColors[15].w
                   + 0.125*myChilds[2].childColors[21].w
                   + 0.0625*myChilds[0].childColors[12].w
                   + 0.0625*myChilds[0].childColors[16].w
                   + 0.0625*myChilds[0].childColors[22].w
                   + 0.0625*myChilds[1].childColors[12].w
                   + 0.0625*myChilds[1].childColors[16].w
                   + 0.0625*myChilds[3].childColors[12].w
                   + 0.0625*myChilds[2].childColors[12].w
                   + 0.0625*myChilds[2].childColors[22].w
                   + 0.03125*myChilds[0].childColors[13].w
                   + 0.03125*myChilds[1].childColors[13].w
                   + 0.03125*myChilds[3].childColors[13].w
                   + 0.03125*myChilds[2].childColors[13].w;

    frontFace.x /= 1.5;
    frontFace.y /= 1.5;
    frontFace.z /= 1.5;
    frontFace.w /= 1.5;

    // BACK FACE
    float4 backFace = make_float4(0,0,0,0);

    // needed in child 4: 221 121 211 111
    // needed in child 5: 121 111
    // needed in child 6: 211 111
    // needed in child 7: 111
    // keep in mind that the backface need less lookups as the faces intersect
    backFace.x += 0.125*myChilds[4].childColors[25].x
                   + 0.0625*myChilds[4].childColors[16].x
                   + 0.0625*myChilds[4].childColors[22].x
                   + 0.0625*myChilds[5].childColors[16].x
                   + 0.0625*myChilds[6].childColors[22].x
                   + 0.03125*myChilds[4].childColors[13].x
                   + 0.03125*myChilds[5].childColors[13].x
                   + 0.03125*myChilds[6].childColors[13].x
                   + 0.03125*myChilds[7].childColors[13].x;

    backFace.y += 0.125*myChilds[4].childColors[25].y
                  + 0.0625*myChilds[4].childColors[16].y
                  + 0.0625*myChilds[4].childColors[22].y
                  + 0.0625*myChilds[5].childColors[16].y
                  + 0.0625*myChilds[6].childColors[22].y
                  + 0.03125*myChilds[4].childColors[13].y
                  + 0.03125*myChilds[5].childColors[13].y
                  + 0.03125*myChilds[6].childColors[13].y
                  + 0.03125*myChilds[7].childColors[13].y;

    backFace.z += 0.125*myChilds[4].childColors[25].z
                  + 0.0625*myChilds[4].childColors[16].z
                  + 0.0625*myChilds[4].childColors[22].z
                  + 0.0625*myChilds[5].childColors[16].z
                  + 0.0625*myChilds[6].childColors[22].z
                  + 0.03125*myChilds[4].childColors[13].z
                  + 0.03125*myChilds[5].childColors[13].z
                  + 0.03125*myChilds[6].childColors[13].z
                  + 0.03125*myChilds[7].childColors[13].z;

    backFace.w += 0.125*myChilds[4].childColors[25].w
                  + 0.0625*myChilds[4].childColors[16].w
                  + 0.0625*myChilds[4].childColors[22].w
                  + 0.0625*myChilds[5].childColors[16].w
                  + 0.0625*myChilds[6].childColors[22].w
                  + 0.03125*myChilds[4].childColors[13].w
                  + 0.03125*myChilds[5].childColors[13].w
                  + 0.03125*myChilds[6].childColors[13].w
                  + 0.03125*myChilds[7].childColors[13].w;

    backFace.x /= 0.5;
    backFace.y /= 0.5;
    backFace.z /= 0.5;
    backFace.w /= 0.5;

    // LEFT face
    // needed ID 0: 0.25*022 0.125*(012 021 122) 0.0625*(121 011 112) 0.03125(111)
    // needed ID 4: 0.125*(021) 0.0625*(011 121) 0.03125(111)
    // needed ID 2: 0.125*(012) 0.0625*(011 112) 0.03125(111)
    // needed ID 6: 0,0625*(011) 0.03125*(111)
    float4 leftFace = make_float4(0,0,0,0);

    leftFace.x += 0.25*myChilds[0].childColors[8].x
                  + 0.125*myChilds[0].childColors[5].x
                  + 0.125*myChilds[0].childColors[7].x
                  + 0.125*myChilds[0].childColors[17].x
                  + 0.125*myChilds[4].childColors[7].x
                  + 0.125*myChilds[2].childColors[5].x
                  + 0.0625*myChilds[0].childColors[16].x
                  + 0.0625*myChilds[0].childColors[4].x
                  + 0.0625*myChilds[0].childColors[14].x
                  + 0.0625*myChilds[4].childColors[4].x
                  + 0.0625*myChilds[4].childColors[16].x
                  + 0.0625*myChilds[2].childColors[4].x
                  + 0.0625*myChilds[2].childColors[14].x
                  + 0.0625*myChilds[6].childColors[4].x
                  + 0.03125*myChilds[0].childColors[13].x
                  + 0.03125*myChilds[4].childColors[13].x
                  + 0.03125*myChilds[2].childColors[13].x
                  + 0.03125*myChilds[6].childColors[13].x;

    leftFace.y += 0.25*myChilds[0].childColors[8].y
                  + 0.125*myChilds[0].childColors[5].y
                  + 0.125*myChilds[0].childColors[7].y
                  + 0.125*myChilds[0].childColors[17].y
                  + 0.125*myChilds[4].childColors[7].y
                  + 0.125*myChilds[2].childColors[5].y
                  + 0.0625*myChilds[0].childColors[16].y
                  + 0.0625*myChilds[0].childColors[4].y
                  + 0.0625*myChilds[0].childColors[14].y
                  + 0.0625*myChilds[4].childColors[4].y
                  + 0.0625*myChilds[4].childColors[16].y
                  + 0.0625*myChilds[2].childColors[4].y
                  + 0.0625*myChilds[2].childColors[14].y
                  + 0.0625*myChilds[6].childColors[4].x
                  + 0.03125*myChilds[0].childColors[13].y
                  + 0.03125*myChilds[4].childColors[13].y
                  + 0.03125*myChilds[2].childColors[13].y
                  + 0.03125*myChilds[6].childColors[13].y;

    leftFace.z += 0.25*myChilds[0].childColors[8].z
                  + 0.125*myChilds[0].childColors[5].z
                  + 0.125*myChilds[0].childColors[7].z
                  + 0.125*myChilds[0].childColors[17].z
                  + 0.125*myChilds[4].childColors[7].z
                  + 0.125*myChilds[2].childColors[5].z
                  + 0.0625*myChilds[0].childColors[16].z
                  + 0.0625*myChilds[0].childColors[4].z
                  + 0.0625*myChilds[0].childColors[14].z
                  + 0.0625*myChilds[4].childColors[4].z
                  + 0.0625*myChilds[4].childColors[16].z
                  + 0.0625*myChilds[2].childColors[4].z
                  + 0.0625*myChilds[2].childColors[14].z
                  + 0.0625*myChilds[6].childColors[4].x
                  + 0.03125*myChilds[0].childColors[13].z
                  + 0.03125*myChilds[4].childColors[13].z
                  + 0.03125*myChilds[2].childColors[13].z
                  + 0.03125*myChilds[6].childColors[13].z;

    leftFace.w += 0.25*myChilds[0].childColors[8].w
                  + 0.125*myChilds[0].childColors[5].w
                  + 0.125*myChilds[0].childColors[7].w
                  + 0.125*myChilds[0].childColors[17].w
                  + 0.125*myChilds[4].childColors[7].w
                  + 0.125*myChilds[2].childColors[5].w
                  + 0.0625*myChilds[0].childColors[16].w
                  + 0.0625*myChilds[0].childColors[4].w
                  + 0.0625*myChilds[0].childColors[14].w
                  + 0.0625*myChilds[4].childColors[4].w
                  + 0.0625*myChilds[4].childColors[16].w
                  + 0.0625*myChilds[2].childColors[4].w
                  + 0.0625*myChilds[2].childColors[14].w
                  + 0.0625*myChilds[6].childColors[4].x
                  + 0.03125*myChilds[0].childColors[13].w
                  + 0.03125*myChilds[4].childColors[13].w
                  + 0.03125*myChilds[2].childColors[13].w
                  + 0.03125*myChilds[6].childColors[13].w;

    leftFace.x /= 1.5;
    leftFace.y /= 1.5;
    leftFace.z /= 1.5;
    leftFace.w /= 1.5;

    // RIGHT FACE
    // needed id 1: 0.125(122) 0.0625(121 112) 0.03125(111)
    // needed id 3: 0.0625(112) 0.03125(111)
    // needed id 5: 0.0625(121) 0.03125(111)
    // needed id 7: 0.03125(111)

    float4 rightFace = make_float4(0,0,0,0);

    rightFace.x += 0.125*myChilds[1].childColors[17].x
                   + 0.0625*myChilds[1].childColors[16].x
                   + 0.0625*myChilds[1].childColors[14].x
                   + 0.0625*myChilds[3].childColors[14].x
                   + 0.0625*myChilds[5].childColors[16].x
                   + 0.03125*myChilds[1].childColors[13].x
                   + 0.03125*myChilds[3].childColors[13].x
                   + 0.03125*myChilds[5].childColors[13].x
                   + 0.03125*myChilds[7].childColors[13].x;

    rightFace.y += 0.125*myChilds[1].childColors[17].y
                   + 0.0625*myChilds[1].childColors[16].y
                   + 0.0625*myChilds[1].childColors[14].y
                   + 0.0625*myChilds[3].childColors[14].y
                   + 0.0625*myChilds[5].childColors[16].y
                   + 0.03125*myChilds[1].childColors[13].y
                   + 0.03125*myChilds[3].childColors[13].y
                   + 0.03125*myChilds[5].childColors[13].y
                   + 0.03125*myChilds[7].childColors[13].y;

    rightFace.z += 0.125*myChilds[1].childColors[17].z
                   + 0.0625*myChilds[1].childColors[16].z
                   + 0.0625*myChilds[1].childColors[14].z
                   + 0.0625*myChilds[3].childColors[14].z
                   + 0.0625*myChilds[5].childColors[16].z
                   + 0.03125*myChilds[1].childColors[13].z
                   + 0.03125*myChilds[3].childColors[13].z
                   + 0.03125*myChilds[5].childColors[13].z
                   + 0.03125*myChilds[7].childColors[13].z;

    rightFace.w += 0.125*myChilds[1].childColors[17].w
                   + 0.0625*myChilds[1].childColors[16].w
                   + 0.0625*myChilds[1].childColors[14].w
                   + 0.0625*myChilds[3].childColors[14].w
                   + 0.0625*myChilds[5].childColors[16].w
                   + 0.03125*myChilds[1].childColors[13].w
                   + 0.03125*myChilds[3].childColors[13].w
                   + 0.03125*myChilds[5].childColors[13].w
                   + 0.03125*myChilds[7].childColors[13].w;

    rightFace.x /= 0.5;
    rightFace.y /= 0.5;
    rightFace.z /= 0.5;
    rightFace.w /= 0.5;

    // TOP FACE
    // needed id 0: 0.25(202) 0.125(201 102 212) 0.0625(112 211 101) 0.03125(111)
    // needed id 1: 0.125(102) 0.0625(112 101) 0.03125(111)
    // needed id 4: 0.125(201) 0.0625(101 211) 0.03125(111)
    // needed id 5: 0.0625(101) 0.03125(111)

    float4 topFace = make_float4(0,0,0,0);

    topFace.x += 0.25*myChilds[0].childColors[20].x
                 + 0.125*myChilds[0].childColors[19].x
                 + 0.125*myChilds[0].childColors[11].x
                 + 0.125*myChilds[0].childColors[23].x
                 + 0.125*myChilds[1].childColors[11].x
                 + 0.125*myChilds[4].childColors[19].x
                 + 0.0625*myChilds[0].childColors[14].x
                 + 0.0625*myChilds[0].childColors[22].x
                 + 0.0625*myChilds[0].childColors[10].x
                 + 0.0625*myChilds[1].childColors[14].x
                 + 0.0625*myChilds[1].childColors[10].x
                 + 0.0625*myChilds[4].childColors[10].x
                 + 0.0625*myChilds[4].childColors[22].x
                 + 0.0625*myChilds[5].childColors[10].x
                 + 0.03125*myChilds[0].childColors[13].x
                 + 0.03125*myChilds[1].childColors[13].x
                 + 0.03125*myChilds[4].childColors[13].x
                 + 0.03125*myChilds[5].childColors[13].x;

    topFace.y += 0.25*myChilds[0].childColors[20].y
                 + 0.125*myChilds[0].childColors[19].y
                 + 0.125*myChilds[0].childColors[11].y
                 + 0.125*myChilds[0].childColors[23].y
                 + 0.125*myChilds[1].childColors[11].y
                 + 0.125*myChilds[4].childColors[19].y
                 + 0.0625*myChilds[0].childColors[14].y
                 + 0.0625*myChilds[0].childColors[22].y
                 + 0.0625*myChilds[0].childColors[10].y
                 + 0.0625*myChilds[1].childColors[14].y
                 + 0.0625*myChilds[1].childColors[10].y
                 + 0.0625*myChilds[4].childColors[10].y
                 + 0.0625*myChilds[4].childColors[22].y
                 + 0.0625*myChilds[5].childColors[10].y
                 + 0.03125*myChilds[0].childColors[13].y
                 + 0.03125*myChilds[1].childColors[13].y
                 + 0.03125*myChilds[4].childColors[13].y
                 + 0.03125*myChilds[5].childColors[13].y;

    topFace.z += 0.25*myChilds[0].childColors[20].z
                 + 0.125*myChilds[0].childColors[19].z
                 + 0.125*myChilds[0].childColors[11].z
                 + 0.125*myChilds[0].childColors[23].z
                 + 0.125*myChilds[1].childColors[11].z
                 + 0.125*myChilds[4].childColors[19].z
                 + 0.0625*myChilds[0].childColors[14].z
                 + 0.0625*myChilds[0].childColors[22].z
                 + 0.0625*myChilds[0].childColors[10].z
                 + 0.0625*myChilds[1].childColors[14].z
                 + 0.0625*myChilds[1].childColors[10].z
                 + 0.0625*myChilds[4].childColors[10].z
                 + 0.0625*myChilds[4].childColors[22].z
                 + 0.0625*myChilds[5].childColors[10].z
                 + 0.03125*myChilds[0].childColors[13].z
                 + 0.03125*myChilds[1].childColors[13].z
                 + 0.03125*myChilds[4].childColors[13].z
                 + 0.03125*myChilds[5].childColors[13].z;

    topFace.w += 0.25*myChilds[0].childColors[20].w
                 + 0.125*myChilds[0].childColors[19].w
                 + 0.125*myChilds[0].childColors[11].w
                 + 0.125*myChilds[0].childColors[23].w
                 + 0.125*myChilds[1].childColors[11].w
                 + 0.125*myChilds[4].childColors[19].w
                 + 0.0625*myChilds[0].childColors[14].w
                 + 0.0625*myChilds[0].childColors[22].w
                 + 0.0625*myChilds[0].childColors[10].w
                 + 0.0625*myChilds[1].childColors[14].w
                 + 0.0625*myChilds[1].childColors[10].w
                 + 0.0625*myChilds[4].childColors[10].w
                 + 0.0625*myChilds[4].childColors[22].w
                 + 0.0625*myChilds[5].childColors[10].w
                 + 0.03125*myChilds[0].childColors[13].w
                 + 0.03125*myChilds[1].childColors[13].w
                 + 0.03125*myChilds[4].childColors[13].w
                 + 0.03125*myChilds[5].childColors[13].w;

    topFace.x /= 1.5;
    topFace.y /= 1.5;
    topFace.z /= 1.5;
    topFace.w /= 1.5;

    // BOTTOM FACE
    // needed ID 2: 0.125*(212) 0.0625*(211 112) 0.03125*(111)
    // needed ID 3: 0.0625*(102) 0.03125*(111)
    // needed ID 6: 0.0625*(221) 0.03125*(111)
    // needed ID 7: 0.03125(111)

    float4 bottomFace = make_float4(0,0,0,0);

    bottomFace.x += 0.125*myChilds[2].childColors[23].x
                    + 0.0625*myChilds[2].childColors[22].x
                    + 0.0625*myChilds[2].childColors[14].x
                    + 0.0625*myChilds[3].childColors[11].x
                    + 0.0625*myChilds[6].childColors[25].x
                    + 0.03125*myChilds[2].childColors[13].x
                    + 0.03125*myChilds[3].childColors[13].x
                    + 0.03125*myChilds[6].childColors[13].x
                    + 0.03125*myChilds[7].childColors[13].x;

    bottomFace.y += 0.125*myChilds[2].childColors[23].y
                    + 0.0625*myChilds[2].childColors[22].y
                    + 0.0625*myChilds[2].childColors[14].y
                    + 0.0625*myChilds[3].childColors[11].y
                    + 0.0625*myChilds[6].childColors[25].y
                    + 0.03125*myChilds[2].childColors[13].y
                    + 0.03125*myChilds[3].childColors[13].y
                    + 0.03125*myChilds[6].childColors[13].y
                    + 0.03125*myChilds[7].childColors[13].y;

    bottomFace.z += 0.125*myChilds[2].childColors[23].z
                    + 0.0625*myChilds[2].childColors[22].z
                    + 0.0625*myChilds[2].childColors[14].z
                    + 0.0625*myChilds[3].childColors[11].z
                    + 0.0625*myChilds[6].childColors[25].z
                    + 0.03125*myChilds[2].childColors[13].z
                    + 0.03125*myChilds[3].childColors[13].z
                    + 0.03125*myChilds[6].childColors[13].z
                    + 0.03125*myChilds[7].childColors[13].z;

    bottomFace.w += 0.125*myChilds[2].childColors[23].w
                    + 0.0625*myChilds[2].childColors[22].w
                    + 0.0625*myChilds[2].childColors[14].w
                    + 0.0625*myChilds[3].childColors[11].w
                    + 0.0625*myChilds[6].childColors[25].w
                    + 0.03125*myChilds[2].childColors[13].w
                    + 0.03125*myChilds[3].childColors[13].w
                    + 0.03125*myChilds[6].childColors[13].w
                    + 0.03125*myChilds[7].childColors[13].w;

    bottomFace.x /= 0.5;
    bottomFace.y /= 0.5;
    bottomFace.z /= 0.5;
    bottomFace.w /= 0.5;


    centerColor.w = 255;
    // center (1,1,1)
    surf3Dwrite(make_uchar4(centerColor.x,centerColor.y,centerColor.z,centerColor.w),
                colorBrickPool,
                (targetBrick.x + 1) * sizeof(uchar4),
                targetBrick.y + 1,
                targetBrick.z + 1);

    leftTopNear.w = 255;
    //0,0,0 leftTopNear // seems ok
    surf3Dwrite(make_uchar4(leftTopNear.x,leftTopNear.y,leftTopNear.z,leftTopNear.w),
                colorBrickPool,
                (targetBrick.x) * sizeof(uchar4),
                targetBrick.y,
                targetBrick.z);

    // 0,0,1
    surf3Dwrite(make_uchar4(centerColor.x,centerColor.y,centerColor.z,centerColor.w),
                colorBrickPool,
                (targetBrick.x) * sizeof(uchar4),
                targetBrick.y,
                targetBrick.z+1);

    leftTopFar.w = 255;
    // 0,0,2 leftTopFar // seems to work
    surf3Dwrite(make_uchar4(leftTopFar.x,leftTopFar.y,leftTopFar.z,leftTopFar.w),
                colorBrickPool,
                (targetBrick.x) * sizeof(uchar4),
                targetBrick.y,
                targetBrick.z+2);

    // 0,1,0
    surf3Dwrite(make_uchar4(centerColor.x,centerColor.y,centerColor.z,centerColor.w),
                colorBrickPool,
                (targetBrick.x) * sizeof(uchar4),
                targetBrick.y+1,
                targetBrick.z);

    leftFace.w = 255;
    // 0,1,1
    surf3Dwrite(make_uchar4(leftFace.x,leftFace.y,leftFace.z,leftFace.w),
                colorBrickPool,
                (targetBrick.x) * sizeof(uchar4),
                targetBrick.y+1,
                targetBrick.z+1);

    // 0,1,2
    surf3Dwrite(make_uchar4(centerColor.x,centerColor.y,centerColor.z,centerColor.w),
                colorBrickPool,
                (targetBrick.x) * sizeof(uchar4),
                targetBrick.y+1,
                targetBrick.z+2);

    leftBottomNear.w = 255;
    // 0,2,0 leftBottomNear // seems to work
    surf3Dwrite(make_uchar4(leftBottomNear.x,leftBottomNear.y,leftBottomNear.z,leftBottomNear.w),
                colorBrickPool,
                (targetBrick.x) * sizeof(uchar4),
                targetBrick.y+2,
                targetBrick.z);

    // 0,2,1
    surf3Dwrite(make_uchar4(centerColor.x,centerColor.y,centerColor.z,centerColor.w),
                colorBrickPool,
                (targetBrick.x) * sizeof(uchar4),
                targetBrick.y+2,
                targetBrick.z+1);

    // 0,2,2 leftBottomFar // seems to work
    leftBottomNear.w = 255;
    surf3Dwrite(make_uchar4(leftBottomFar.x,leftBottomFar.y,leftBottomFar.z,leftBottomFar.w),
                colorBrickPool,
                (targetBrick.x) * sizeof(uchar4),
                targetBrick.y+2,
                targetBrick.z+2);

    // 1,0,0
    surf3Dwrite(make_uchar4(centerColor.x,centerColor.y,centerColor.z,centerColor.w),
                colorBrickPool,
                (targetBrick.x+1) * sizeof(uchar4),
                targetBrick.y,
                targetBrick.z);
    // 1,0,1
    topFace.w = 255;
    surf3Dwrite(make_uchar4(topFace.x,topFace.y,topFace.z,topFace.w),
                colorBrickPool,
                (targetBrick.x+1) * sizeof(uchar4),
                targetBrick.y,
                targetBrick.z+1);

    // 1,0,2
    surf3Dwrite(make_uchar4(centerColor.x,centerColor.y,centerColor.z,centerColor.w),
                colorBrickPool,
                (targetBrick.x+1) * sizeof(uchar4),
                targetBrick.y,
                targetBrick.z+2);

    // 1,1,0
    frontFace.w = 255;
    surf3Dwrite(make_uchar4(frontFace.x,frontFace.y,frontFace.z,frontFace.w),
                colorBrickPool,
                (targetBrick.x+1) * sizeof(uchar4),
                targetBrick.y+1,
                targetBrick.z);

    // 1,1,2
    backFace.w = 255;
    surf3Dwrite(make_uchar4(backFace.x,backFace.y,backFace.z,backFace.w),
                colorBrickPool,
                (targetBrick.x+1) * sizeof(uchar4),
                targetBrick.y+1,
                targetBrick.z+2);

    // 1,2,0
    surf3Dwrite(make_uchar4(centerColor.x,centerColor.y,centerColor.z,centerColor.w),
                colorBrickPool,
                (targetBrick.x+1) * sizeof(uchar4),
                targetBrick.y+2,
                targetBrick.z);

    // 1,2,1
    bottomFace.w = 255;
    surf3Dwrite(make_uchar4(bottomFace.x,bottomFace.y,bottomFace.z,bottomFace.w),
                colorBrickPool,
                (targetBrick.x+1) * sizeof(uchar4),
                targetBrick.y+2,
                targetBrick.z+1);

    // 1,2,2
    surf3Dwrite(make_uchar4(centerColor.x,centerColor.y,centerColor.z,centerColor.w),
                colorBrickPool,
                (targetBrick.x+1) * sizeof(uchar4),
                targetBrick.y+2,
                targetBrick.z+2);

    // 2,0,0 rightTopNear // seems to work
    rightTopNear.w = 255;
    surf3Dwrite(make_uchar4(rightTopNear.x,rightTopNear.y,rightTopNear.z,rightTopNear.w),
                colorBrickPool,
                (targetBrick.x+2) * sizeof(uchar4),
                targetBrick.y,
                targetBrick.z);

    // 2,0,1
    surf3Dwrite(make_uchar4(centerColor.x,centerColor.y,centerColor.z,centerColor.w),
                colorBrickPool,
                (targetBrick.x+2) * sizeof(uchar4),
                targetBrick.y,
                targetBrick.z+1);

    // 2,0,2 rightTopFar // might be empty?
    rightTopFar.w = 255;
    surf3Dwrite(make_uchar4(rightTopFar.x,rightTopFar.y,rightTopFar.z,rightTopFar.w),
                colorBrickPool,
                (targetBrick.x+2) * sizeof(uchar4),
                targetBrick.y,
                targetBrick.z+2);

    // 2,1,0
    surf3Dwrite(make_uchar4(centerColor.x,centerColor.y,centerColor.z,centerColor.w),
                colorBrickPool,
                (targetBrick.x+2) * sizeof(uchar4),
                targetBrick.y+1,
                targetBrick.z);

    // 2,1,1
    rightFace.w = 255;
    surf3Dwrite(make_uchar4(rightFace.x,rightFace.y,rightFace.z,rightFace.w),
                colorBrickPool,
                (targetBrick.x+2) * sizeof(uchar4),
                targetBrick.y+1,
                targetBrick.z+1);

    // 2,1,2
    surf3Dwrite(make_uchar4(centerColor.x,centerColor.y,centerColor.z,centerColor.w),
                colorBrickPool,
                (targetBrick.x+2) * sizeof(uchar4),
                targetBrick.y+1,
                targetBrick.z+2);

    // 2,2,0 rightBottomNear // seems to work
    rightBottomNear.w = 255;
    surf3Dwrite(make_uchar4(rightBottomNear.x,rightBottomNear.y,rightBottomNear.z,rightBottomNear.w),
                colorBrickPool,
                (targetBrick.x+2) * sizeof(uchar4),
                targetBrick.y+2,
                targetBrick.z);

    // 2,2,1
    surf3Dwrite(make_uchar4(centerColor.x,centerColor.y,centerColor.z,centerColor.w),
                colorBrickPool,
                (targetBrick.x+2) * sizeof(uchar4),
                targetBrick.y+2,
                targetBrick.z+1);

    // 2,2,2 rightBottomFar // we have a winner it does not work
    rightBottomFar.w = 255;
    surf3Dwrite(make_uchar4(rightBottomFar.x,rightBottomFar.y,rightBottomFar.z,rightBottomFar.w),
                colorBrickPool,
                (targetBrick.x+2) * sizeof(uchar4),
                targetBrick.y+2,
                targetBrick.z+2);

}

__global__
void mipMapOctreeLevel(node *nodePool, unsigned int level)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    // make sure our index matches the node-adresses in a given octree level
    index += (constLevelIntervalMap[level].start)*8;
    // make sure we dont load invalid adresses
    if(index >= (constLevelIntervalMap[level].end)*8)
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
            brickCoords[i] = decodeBrickCoords(nodePool[childPointer*8+i].value);
        }

        // finally mipmap our node
        mipMapIsotropic(targetBrick,brickCoords);
        setBit(nodePool[index].value,31);
    }
}

__global__
void combineBrickBordersFast(node *nodePool, neighbours* neighbourPool, unsigned int level)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    // make sure our index matches the node-adresses in a given octree level
    index += (constLevelIntervalMap[level].start)*8;
    // make sure we dont load invalid adresses
    if(index >= (constLevelIntervalMap[level].end)*8)
        return;

    // load the target node that should be filled by mipmapping
    node targetNode = nodePool[index];

    if((getBit(targetNode.nodeTilePointer,32) == 0 && level == 6) || (getBit(targetNode.nodeTilePointer,32) == 1 && level < 6))
    {
        neighbours targetNeighbours = neighbourPool[index];

        // here we have our brick
        uint3 brickCoords = decodeBrickCoords(targetNode.value);

        uchar4 myColors[9];
        uchar4 neighbourColors[9];

        for (int i = 0; i < 9; i++) {
            myColors[i] = make_uchar4(0, 0, 0, 0);
            neighbourColors[i] = make_uchar4(0, 0, 0, 0);
        }

        // load all 6 neighbours
        uint3 nXbrickCoords = decodeBrickCoords(nodePool[targetNeighbours.X].value);
        uint3 nYbrickCoords = decodeBrickCoords(nodePool[targetNeighbours.Y].value);
        uint3 nZbrickCoords = decodeBrickCoords(nodePool[targetNeighbours.Z].value);
        uint3 nNegXbrickCoords = decodeBrickCoords(nodePool[targetNeighbours.negX].value);
        uint3 nNegYbrickCoords = decodeBrickCoords(nodePool[targetNeighbours.negY].value);
        uint3 nNegZbrickCoords = decodeBrickCoords(nodePool[targetNeighbours.negZ].value);


        if (targetNeighbours.Y != 0) {
            // TOP
            surf3Dread(&myColors[0], colorBrickPool, (0 + brickCoords.x) * sizeof(uchar4), 0 + brickCoords.y,
                       0 + brickCoords.z);
            surf3Dread(&neighbourColors[0], colorBrickPool, (0 + nYbrickCoords.x) * sizeof(uchar4),
                       2 + nYbrickCoords.y, 0 + nYbrickCoords.z);

            surf3Dread(&myColors[1], colorBrickPool, (1 + brickCoords.x) * sizeof(uchar4), 0 + brickCoords.y,
                       0 + brickCoords.z);
            surf3Dread(&neighbourColors[1], colorBrickPool, (1 + nYbrickCoords.x) * sizeof(uchar4),
                       2 + nYbrickCoords.y, 0 + nYbrickCoords.z);

            surf3Dread(&myColors[2], colorBrickPool, (2 + brickCoords.x) * sizeof(uchar4), 0 + brickCoords.y,
                       0 + brickCoords.z);
            surf3Dread(&neighbourColors[2], colorBrickPool, (2 + nYbrickCoords.x) * sizeof(uchar4),
                       2 + nYbrickCoords.y, 0 + nYbrickCoords.z);

            surf3Dread(&myColors[3], colorBrickPool, (0 + brickCoords.x) * sizeof(uchar4), 0 + brickCoords.y,
                       1 + brickCoords.z);
            surf3Dread(&neighbourColors[3], colorBrickPool, (0 + nYbrickCoords.x) * sizeof(uchar4),
                       2 + nYbrickCoords.y, 1 + nYbrickCoords.z);


            surf3Dread(&myColors[4], colorBrickPool, (1 + brickCoords.x) * sizeof(uchar4), 0 + brickCoords.y,
                       1 + brickCoords.z);
            surf3Dread(&neighbourColors[4], colorBrickPool, (1 + nYbrickCoords.x) * sizeof(uchar4),
                       2 + nYbrickCoords.y, 1 + nYbrickCoords.z);

            surf3Dread(&myColors[5], colorBrickPool, (2 + brickCoords.x) * sizeof(uchar4), 0 + brickCoords.y,
                       1 + brickCoords.z);
            surf3Dread(&neighbourColors[5], colorBrickPool, (2 + nYbrickCoords.x) * sizeof(uchar4),
                      2 + nYbrickCoords.y, 1 + nYbrickCoords.z);

            surf3Dread(&myColors[6], colorBrickPool, (0 + brickCoords.x) * sizeof(uchar4), 0 + brickCoords.y,
                       2 + brickCoords.z);
            surf3Dread(&neighbourColors[6], colorBrickPool, (0 + nYbrickCoords.x) * sizeof(uchar4),
                       2 + nYbrickCoords.y, 2 + nYbrickCoords.z);

            surf3Dread(&myColors[7], colorBrickPool, (1 + brickCoords.x) * sizeof(uchar4), 0 + brickCoords.y,
                       2 + brickCoords.z);
            surf3Dread(&neighbourColors[7], colorBrickPool, (1 + nYbrickCoords.x) * sizeof(uchar4),
                       2 + nYbrickCoords.y, 2 + nYbrickCoords.z);

            surf3Dread(&myColors[8], colorBrickPool, (2 + brickCoords.x) * sizeof(uchar4), 0 + brickCoords.y,
                       2 + brickCoords.z);
            surf3Dread(&neighbourColors[8], colorBrickPool, (2 + nYbrickCoords.x) * sizeof(uchar4),
                       2 + nYbrickCoords.y, 2 + nYbrickCoords.z);
/*
            for(int i=0;i<9;i++)
                printf("color: %d %d %d index %d color: %d\n", myColors[i].x, myColors[i].y, myColors[i].z, index, i);*/


            __syncthreads();

            //CORNER
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

            // CORNER  => HIER GEHT IRGENDWAS SCHIEF. Wenn man myColors und neighbourcolors ohne zu mitteln schreibt, dann snid die Vorhänge besser
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
                        1 + brickCoords.z);
            surf3Dwrite(tmp, colorBrickPool, (2 + nYbrickCoords.x) * sizeof(uchar4), 2 + nYbrickCoords.y,
                        1 + nYbrickCoords.z);

            //CORNER
            tmp = avgColor(myColors[6], neighbourColors[6]);
            surf3Dwrite(tmp, colorBrickPool, (0 + brickCoords.x) * sizeof(uchar4), 0 + brickCoords.y,
                        2 + brickCoords.z);
            surf3Dwrite(tmp, colorBrickPool, (0 + nYbrickCoords.x) * sizeof(uchar4), 2 + nYbrickCoords.y,
                        2 + nYbrickCoords.z);

            tmp = avgColor(myColors[7], neighbourColors[7]);
            surf3Dwrite(tmp, colorBrickPool, (1 + brickCoords.x) * sizeof(uchar4), 0 + brickCoords.y,
                        2 + brickCoords.z);
            surf3Dwrite(tmp, colorBrickPool, (1 + nYbrickCoords.x) * sizeof(uchar4), 2 + nYbrickCoords.y,
                        2 + nYbrickCoords.z);

            // CORNER
            tmp = avgColor(myColors[8], neighbourColors[8]);
            surf3Dwrite(tmp, colorBrickPool, (2 + brickCoords.x) * sizeof(uchar4), 0 + brickCoords.y,
                        2 + brickCoords.z);
            surf3Dwrite(tmp, colorBrickPool, (2 + nYbrickCoords.x) * sizeof(uchar4), 2 + nYbrickCoords.y,
                        2 + nYbrickCoords.z);
        }
    }
}

#endif