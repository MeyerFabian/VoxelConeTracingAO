#ifndef BRICK_UTILITIES_CUH
#define BRICK_UTILITIES_CUH

//#include "globalResources.cuh"

#include <vector_types.h>

// converts a 1D index (coming from the global brick counter) to a 3D index within the range [0...brickPoolSideLength-1] for each axis
// the brickSize is considered to prevent overlapping bricks
__device__ uint3 getBrickCoords(unsigned int brickAdress, unsigned int brickPoolSideLength, unsigned int brickSideLength = 3)
{
    uint3 coords;
    brickPoolSideLength /=brickSideLength; // make sure that the brick size is considered

    // calculates 1D to 3D index
    coords.x = brickAdress / (brickPoolSideLength*brickPoolSideLength);
    coords.y = (brickAdress / brickPoolSideLength) % brickPoolSideLength;
    coords.z = brickAdress % brickPoolSideLength;

    // bricksidelength as offset (prevents overlapping bricks within the brick pool)
    coords.x = coords.x*brickSideLength;
    coords.y = coords.y*brickSideLength;
    coords.z = coords.z*brickSideLength;

    return coords;
}

// encodes brick coordinates (x,y,z) in a single unsigned integer (10 bits for each coordinate)
// 00 0000000000 0000000000 0000000000
//     X-coord    Y-coord    Z-coord
__device__ unsigned int encodeBrickCoords(uint3 coords)
{
unsigned int codeX = ((0x000003FF & coords.x) << 20U);
unsigned int codeY = ((0x000003FF & coords.y) << 10U);
unsigned int codeZ = ((0x000003FF & coords.z));
unsigned int code = codeX | codeY | codeZ;

return code;
}
// decode brick coordinates that are decoded with the encodeBrickCoords() method
__device__ uint3 decodeBrickCoords(unsigned int coded)
{
    uint3 coords;
    coords.z = coded & 0x000003FF;
    coords.y = (coded & 0x000FFC00) >> 10U;
    coords.x = (coded & 0x3FF00000) >> 20U;
    return coords;
}

// fills the corners of a given brick with a voxel and its color
__device__ void fillBrickCorners(const uint3 &brickCoords, const float3 &voxelPosition, const uchar4 &color)
{
    uint3 nextOctant;
    nextOctant.x = static_cast<unsigned int>(2 * voxelPosition.x);
    nextOctant.y = static_cast<unsigned int>(2 * voxelPosition.y);
    nextOctant.z = static_cast<unsigned int>(2 * voxelPosition.z);

    unsigned int offset = nextOctant.x + 2 * nextOctant.y + 4 * nextOctant.z;

    uint3 pos = insertPositions[offset];
    pos.x += brickCoords.x;
    pos.y += brickCoords.y;
    pos.z += brickCoords.z;

    /*
if(pos.z<10) {
    printf("offset : %d\n", offset);
    printf("color r: %d g: %d b: %d, a:%d\n", static_cast<unsigned int>(color.x), color.y, color.z, color.w);
    printf("posX: %d, posY: %d, posZ: %d\n", pos.x, pos.y, pos.z);
}
*/
    // write the color value to the corner TODO: use a shared counter to prevent race conditions between double list entries in the fragment list
    surf3Dwrite(color, colorBrickPool, pos.x* sizeof(uchar4), pos.y, pos.z);
}

// filters the brick with an inverse gaussian mask to fill a brick
// the corner bricks are used as sources
__device__ void filterBrick(const uint3 &brickCoords)
{
	
    uchar4 colors[8];
    colors[0] = make_uchar4(0,0,0,0);
    colors[1] = make_uchar4(0,0,0,0);
    colors[2] = make_uchar4(0,0,0,0);
    colors[3] = make_uchar4(0,0,0,0);
    colors[4] = make_uchar4(0,0,0,0);
    colors[5] = make_uchar4(0,0,0,0);
    colors[6] = make_uchar4(0,0,0,0);
    colors[7] = make_uchar4(0,0,0,0);
	
    // start by loading all 8 corners to gpu registers
    surf3Dread(&colors[0], colorBrickPool, (insertPositions[0].x + brickCoords.x) * sizeof(uchar4), insertPositions[0].y + brickCoords.y, insertPositions[0].z + brickCoords.z);
    surf3Dread(&colors[1], colorBrickPool, (insertPositions[1].x + brickCoords.x) * sizeof(uchar4), insertPositions[1].y + brickCoords.y, insertPositions[1].z + brickCoords.z);
    surf3Dread(&colors[2], colorBrickPool, (insertPositions[2].x + brickCoords.x) * sizeof(uchar4), insertPositions[2].y + brickCoords.y, insertPositions[2].z + brickCoords.z);
    surf3Dread(&colors[3], colorBrickPool, (insertPositions[3].x + brickCoords.x) * sizeof(uchar4), insertPositions[3].y + brickCoords.y, insertPositions[3].z + brickCoords.z);
    surf3Dread(&colors[4], colorBrickPool, (insertPositions[4].x + brickCoords.x) * sizeof(uchar4), insertPositions[4].y + brickCoords.y, insertPositions[4].z + brickCoords.z);
    surf3Dread(&colors[5], colorBrickPool, (insertPositions[5].x + brickCoords.x) * sizeof(uchar4), insertPositions[5].y + brickCoords.y, insertPositions[5].z + brickCoords.z);
    surf3Dread(&colors[6], colorBrickPool, (insertPositions[6].x + brickCoords.x) * sizeof(uchar4), insertPositions[6].y + brickCoords.y, insertPositions[6].z + brickCoords.z);
    surf3Dread(&colors[7], colorBrickPool, (insertPositions[7].x + brickCoords.x) * sizeof(uchar4), insertPositions[7].y + brickCoords.y, insertPositions[7].z + brickCoords.z);
	
    // ################ center: #######################
    float4 tmp = make_float4(0,0,0,0);

    tmp.x += static_cast<float>(colors[0].x);
    tmp.y += static_cast<float>(colors[0].y);
    tmp.z += static_cast<float>(colors[0].z);
    tmp.w += static_cast<float>(colors[0].w);

    tmp.x += static_cast<float>(colors[1].x);
    tmp.y += static_cast<float>(colors[1].y);
    tmp.z += static_cast<float>(colors[1].z);
    tmp.w += static_cast<float>(colors[1].w);

    tmp.x += static_cast<float>(colors[2].x);
    tmp.y += static_cast<float>(colors[2].y);
    tmp.z += static_cast<float>(colors[2].z);
    tmp.w += static_cast<float>(colors[2].w);

    tmp.x += static_cast<float>(colors[3].x);
    tmp.y += static_cast<float>(colors[3].y);
    tmp.z += static_cast<float>(colors[3].z);
    tmp.w += static_cast<float>(colors[3].w);

    tmp.x += static_cast<float>(colors[4].x);
    tmp.y += static_cast<float>(colors[4].y);
    tmp.z += static_cast<float>(colors[4].z);
    tmp.w += static_cast<float>(colors[4].w);

    tmp.x += static_cast<float>(colors[5].x);
    tmp.y += static_cast<float>(colors[5].y);
    tmp.z += static_cast<float>(colors[5].z);
    tmp.w += static_cast<float>(colors[5].w);

    tmp.x += static_cast<float>(colors[6].x);
    tmp.y += static_cast<float>(colors[6].y);
    tmp.z += static_cast<float>(colors[6].z);
    tmp.w += static_cast<float>(colors[6].w);

    tmp.x += static_cast<float>(colors[7].x);
    tmp.y += static_cast<float>(colors[7].y);
    tmp.z += static_cast<float>(colors[7].z);
    tmp.w += static_cast<float>(colors[7].w);

    tmp.x *= 0.125f;
	tmp.y *= 0.125f;
	tmp.z *= 0.125f;
    tmp.w *= 0.125f;

    uint3 newCoords = make_uint3(1,1,1);
    newCoords.x+=brickCoords.x;
    newCoords.y+=brickCoords.y;
    newCoords.z+=brickCoords.z;

    __syncthreads();

    surf3Dwrite(make_uchar4(tmp.x,tmp.y,tmp.z,tmp.w), colorBrickPool, newCoords.x*sizeof(uchar4), newCoords.y, newCoords.z);

    // ################### FACES ##########################
    // right side: 1, 3, 5, 7
    tmp = make_float4(0,0,0,0);
    tmp.x += static_cast<float>(colors[1].x);
    tmp.y += static_cast<float>(colors[1].y);
    tmp.z += static_cast<float>(colors[1].z);
    tmp.w += static_cast<float>(colors[1].w);

    tmp.x += static_cast<float>(colors[3].x);
    tmp.y += static_cast<float>(colors[3].y);
    tmp.z += static_cast<float>(colors[3].z);
    tmp.w += static_cast<float>(colors[3].w);

    tmp.x += static_cast<float>(colors[5].x);
    tmp.y += static_cast<float>(colors[5].y);
    tmp.z += static_cast<float>(colors[5].z);
    tmp.w += static_cast<float>(colors[5].w);

    tmp.x += static_cast<float>(colors[7].x);
    tmp.y += static_cast<float>(colors[7].y);
    tmp.z += static_cast<float>(colors[7].z);
    tmp.w += static_cast<float>(colors[7].w);

    tmp.x *= 0.25f;
    tmp.y *= 0.25f;
    tmp.z *= 0.25f;
    tmp.w *= 0.25f;

    newCoords = make_uint3(2,1,1);
    newCoords.x+=brickCoords.x;
    newCoords.y+=brickCoords.y;
    newCoords.z+=brickCoords.z;

    surf3Dwrite(make_uchar4(tmp.x,tmp.y,tmp.z,tmp.w), colorBrickPool, newCoords.x*sizeof(uchar4), newCoords.y, newCoords.z);

    // left side: 0, 2, 4, 6
    tmp = make_float4(0,0,0,0);
    tmp.x += static_cast<float>(colors[0].x);
    tmp.y += static_cast<float>(colors[0].y);
    tmp.z += static_cast<float>(colors[0].z);
    tmp.w += static_cast<float>(colors[0].w);

    tmp.x += static_cast<float>(colors[2].x);
    tmp.y += static_cast<float>(colors[2].y);
    tmp.z += static_cast<float>(colors[2].z);
    tmp.w += static_cast<float>(colors[2].w);

    tmp.x += static_cast<float>(colors[4].x);
    tmp.y += static_cast<float>(colors[4].y);
    tmp.z += static_cast<float>(colors[4].z);
    tmp.w += static_cast<float>(colors[4].w);

    tmp.x += static_cast<float>(colors[6].x);
    tmp.y += static_cast<float>(colors[6].y);
    tmp.z += static_cast<float>(colors[6].z);
    tmp.w += static_cast<float>(colors[6].w);

    tmp.x *= 0.25f;
    tmp.y *= 0.25f;
    tmp.z *= 0.25f;
    tmp.w *= 0.25f;

    newCoords = make_uint3(0,1,1);
    newCoords.x+=brickCoords.x;
    newCoords.y+=brickCoords.y;
    newCoords.z+=brickCoords.z;

    surf3Dwrite(make_uchar4(tmp.x,tmp.y,tmp.z,tmp.w), colorBrickPool, newCoords.x*sizeof(uchar4), newCoords.y, newCoords.z);

    // bottom side: 2, 3, 6, 7
    tmp = make_float4(0,0,0,0);
    tmp.x += static_cast<float>(colors[2].x);
    tmp.y += static_cast<float>(colors[2].y);
    tmp.z += static_cast<float>(colors[2].z);
    tmp.w += static_cast<float>(colors[2].w);

    tmp.x += static_cast<float>(colors[3].x);
    tmp.y += static_cast<float>(colors[3].y);
    tmp.z += static_cast<float>(colors[3].z);
    tmp.w += static_cast<float>(colors[3].w);

    tmp.x += static_cast<float>(colors[6].x);
    tmp.y += static_cast<float>(colors[6].y);
    tmp.z += static_cast<float>(colors[6].z);
    tmp.w += static_cast<float>(colors[6].w);

    tmp.x += static_cast<float>(colors[7].x);
    tmp.y += static_cast<float>(colors[7].y);
    tmp.z += static_cast<float>(colors[7].z);
    tmp.w += static_cast<float>(colors[7].w);

    tmp.x *= 0.25f;
    tmp.y *= 0.25f;
    tmp.z *= 0.25f;
    tmp.w *= 0.25f;

    newCoords = make_uint3(1,2,1);
    newCoords.x+=brickCoords.x;
    newCoords.y+=brickCoords.y;
    newCoords.z+=brickCoords.z;

    surf3Dwrite(make_uchar4(tmp.x,tmp.y,tmp.z,tmp.w), colorBrickPool, newCoords.x*sizeof(uchar4), newCoords.y, newCoords.z);

    // top side: 0, 1, 4, 5
    tmp = make_float4(0,0,0,0);
    tmp.x += static_cast<float>(colors[0].x);
    tmp.y += static_cast<float>(colors[0].y);
    tmp.z += static_cast<float>(colors[0].z);
    tmp.w += static_cast<float>(colors[0].w);

    tmp.x += static_cast<float>(colors[1].x);
    tmp.y += static_cast<float>(colors[1].y);
    tmp.z += static_cast<float>(colors[1].z);
    tmp.w += static_cast<float>(colors[1].w);

    tmp.x += static_cast<float>(colors[4].x);
    tmp.y += static_cast<float>(colors[4].y);
    tmp.z += static_cast<float>(colors[4].z);
    tmp.w += static_cast<float>(colors[4].w);

    tmp.x += static_cast<float>(colors[5].x);
    tmp.y += static_cast<float>(colors[5].y);
    tmp.z += static_cast<float>(colors[5].z);
    tmp.w += static_cast<float>(colors[5].w);

    tmp.x *= 0.25f;
    tmp.y *= 0.25f;
    tmp.z *= 0.25f;
    tmp.w *= 0.25f;

    newCoords = make_uint3(1,0,1);
    newCoords.x+=brickCoords.x;
    newCoords.y+=brickCoords.y;
    newCoords.z+=brickCoords.z;

    surf3Dwrite(make_uchar4(tmp.x,tmp.y,tmp.z,tmp.w), colorBrickPool, newCoords.x*sizeof(uchar4), newCoords.y, newCoords.z);

    // near side: 0, 1, 2, 3
    tmp = make_float4(0,0,0,0);
    tmp.x += static_cast<float>(colors[0].x);
    tmp.y += static_cast<float>(colors[0].y);
    tmp.z += static_cast<float>(colors[0].z);
    tmp.w += static_cast<float>(colors[0].w);

    tmp.x += static_cast<float>(colors[1].x);
    tmp.y += static_cast<float>(colors[1].y);
    tmp.z += static_cast<float>(colors[1].z);
    tmp.w += static_cast<float>(colors[1].w);

    tmp.x += static_cast<float>(colors[2].x);
    tmp.y += static_cast<float>(colors[2].y);
    tmp.z += static_cast<float>(colors[2].z);
    tmp.w += static_cast<float>(colors[2].w);

    tmp.x += static_cast<float>(colors[3].x);
    tmp.y += static_cast<float>(colors[3].y);
    tmp.z += static_cast<float>(colors[3].z);
    tmp.w += static_cast<float>(colors[3].w);

    tmp.x *= 0.25f;
    tmp.y *= 0.25f;
    tmp.z *= 0.25f;
    tmp.w *= 0.25f;

    newCoords = make_uint3(1,1,0);
    newCoords.x+=brickCoords.x;
    newCoords.y+=brickCoords.y;
    newCoords.z+=brickCoords.z;

    surf3Dwrite(make_uchar4(tmp.x,tmp.y,tmp.z,tmp.w), colorBrickPool, newCoords.x*sizeof(uchar4), newCoords.y, newCoords.z);

    // far side: 4, 5, 6, 7
    tmp = make_float4(0,0,0,0);
    tmp.x += static_cast<float>(colors[4].x);
    tmp.y += static_cast<float>(colors[4].y);
    tmp.z += static_cast<float>(colors[4].z);
    tmp.w += static_cast<float>(colors[4].w);

    tmp.x += static_cast<float>(colors[5].x);
    tmp.y += static_cast<float>(colors[5].y);
    tmp.z += static_cast<float>(colors[5].z);
    tmp.w += static_cast<float>(colors[5].w);

    tmp.x += static_cast<float>(colors[6].x);
    tmp.y += static_cast<float>(colors[6].y);
    tmp.z += static_cast<float>(colors[6].z);
    tmp.w += static_cast<float>(colors[6].w);

    tmp.x += static_cast<float>(colors[7].x);
    tmp.y += static_cast<float>(colors[7].y);
    tmp.z += static_cast<float>(colors[7].z);
    tmp.w += static_cast<float>(colors[7].w);

    tmp.x *= 0.25f;
    tmp.y *= 0.25f;
    tmp.z *= 0.25f;
    tmp.w *= 0.25f;

    newCoords = make_uint3(1,1,2);
    newCoords.x+=brickCoords.x;
    newCoords.y+=brickCoords.y;
    newCoords.z+=brickCoords.z;

    surf3Dwrite(make_uchar4(tmp.x,tmp.y,tmp.z,tmp.w), colorBrickPool, newCoords.x*sizeof(uchar4), newCoords.y, newCoords.z);

    // ####################### EDGES (FRONT) #####################
    // top edge
    tmp = make_float4(0,0,0,0);
    tmp.x += static_cast<float>(colors[0].x);
    tmp.y += static_cast<float>(colors[0].y);
    tmp.z += static_cast<float>(colors[0].z);
    tmp.w += static_cast<float>(colors[0].w);

    tmp.x += static_cast<float>(colors[1].x);
    tmp.y += static_cast<float>(colors[1].y);
    tmp.z += static_cast<float>(colors[1].z);
    tmp.w += static_cast<float>(colors[1].w);

    tmp.x *= 0.5f;
    tmp.y *= 0.5f;
    tmp.z *= 0.5f;
    tmp.w *= 0.5f;

    newCoords = make_uint3(1,0,0);
    newCoords.x+=brickCoords.x;
    newCoords.y+=brickCoords.y;
    newCoords.z+=brickCoords.z;

    surf3Dwrite(make_uchar4(tmp.x,tmp.y,tmp.z,tmp.w), colorBrickPool, newCoords.x*sizeof(uchar4), newCoords.y, newCoords.z);

    // bottom edge
    tmp = make_float4(0,0,0,0);
    tmp.x += static_cast<float>(colors[2].x);
    tmp.y += static_cast<float>(colors[2].y);
    tmp.z += static_cast<float>(colors[2].z);
    tmp.w += static_cast<float>(colors[2].w);

    tmp.x += static_cast<float>(colors[3].x);
    tmp.y += static_cast<float>(colors[3].y);
    tmp.z += static_cast<float>(colors[3].z);
    tmp.w += static_cast<float>(colors[3].w);

    tmp.x *= 0.5f;
    tmp.y *= 0.5f;
    tmp.z *= 0.5f;
    tmp.w *= 0.5f;

    newCoords = make_uint3(1,2,0);
    newCoords.x+=brickCoords.x;
    newCoords.y+=brickCoords.y;
    newCoords.z+=brickCoords.z;

    surf3Dwrite(make_uchar4(tmp.x,tmp.y,tmp.z,tmp.w), colorBrickPool, newCoords.x*sizeof(uchar4), newCoords.y, newCoords.z);

    // left edge
    tmp = make_float4(0,0,0,0);
    tmp.x += static_cast<float>(colors[0].x);
    tmp.y += static_cast<float>(colors[0].y);
    tmp.z += static_cast<float>(colors[0].z);
    tmp.w += static_cast<float>(colors[0].w);

    tmp.x += static_cast<float>(colors[2].x);
    tmp.y += static_cast<float>(colors[2].y);
    tmp.z += static_cast<float>(colors[2].z);
    tmp.w += static_cast<float>(colors[2].w);

    tmp.x *= 0.5f;
    tmp.y *= 0.5f;
    tmp.z *= 0.5f;
    tmp.w *= 0.5f;

    newCoords = make_uint3(0,1,0);
    newCoords.x+=brickCoords.x;
    newCoords.y+=brickCoords.y;
    newCoords.z+=brickCoords.z;

    surf3Dwrite(make_uchar4(tmp.x,tmp.y,tmp.z,tmp.w), colorBrickPool, newCoords.x*sizeof(uchar4), newCoords.y, newCoords.z);

    // right edge
    tmp = make_float4(0,0,0,0);
    tmp.x += static_cast<float>(colors[1].x);
    tmp.y += static_cast<float>(colors[1].y);
    tmp.z += static_cast<float>(colors[1].z);
    tmp.w += static_cast<float>(colors[1].w);

    tmp.x += static_cast<float>(colors[3].x);
    tmp.y += static_cast<float>(colors[3].y);
    tmp.z += static_cast<float>(colors[3].z);
    tmp.w += static_cast<float>(colors[3].w);

    tmp.x *= 0.5f;
    tmp.y *= 0.5f;
    tmp.z *= 0.5f;
    tmp.w *= 0.5f;

    newCoords = make_uint3(2,1,0);
    newCoords.x+=brickCoords.x;
    newCoords.y+=brickCoords.y;
    newCoords.z+=brickCoords.z;

    surf3Dwrite(make_uchar4(tmp.x,tmp.y,tmp.z,tmp.w), colorBrickPool, newCoords.x*sizeof(uchar4), newCoords.y, newCoords.z);

    // ####################### EDGES (BACK) #####################
    // top edge
    tmp = make_float4(0,0,0,0);
    tmp.x += static_cast<float>(colors[4].x);
    tmp.y += static_cast<float>(colors[4].y);
    tmp.z += static_cast<float>(colors[4].z);
    tmp.w += static_cast<float>(colors[4].w);

    tmp.x += static_cast<float>(colors[5].x);
    tmp.y += static_cast<float>(colors[5].y);
    tmp.z += static_cast<float>(colors[5].z);
    tmp.w += static_cast<float>(colors[5].w);

    tmp.x *= 0.5f;
    tmp.y *= 0.5f;
    tmp.z *= 0.5f;
    tmp.w *= 0.5f;

    newCoords = make_uint3(1,0,2);
    newCoords.x+=brickCoords.x;
    newCoords.y+=brickCoords.y;
    newCoords.z+=brickCoords.z;

    surf3Dwrite(make_uchar4(tmp.x,tmp.y,tmp.z,tmp.w), colorBrickPool, newCoords.x*sizeof(uchar4), newCoords.y, newCoords.z);

    // bottom edge
    tmp = make_float4(0,0,0,0);
    tmp.x += static_cast<float>(colors[6].x);
    tmp.y += static_cast<float>(colors[6].y);
    tmp.z += static_cast<float>(colors[6].z);
    tmp.w += static_cast<float>(colors[6].w);

    tmp.x += static_cast<float>(colors[7].x);
    tmp.y += static_cast<float>(colors[7].y);
    tmp.z += static_cast<float>(colors[7].z);
    tmp.w += static_cast<float>(colors[7].w);

    tmp.x *= 0.5f;
    tmp.y *= 0.5f;
    tmp.z *= 0.5f;
    tmp.w *= 0.5f;

    newCoords = make_uint3(1,2,2);
    newCoords.x+=brickCoords.x;
    newCoords.y+=brickCoords.y;
    newCoords.z+=brickCoords.z;

    surf3Dwrite(make_uchar4(tmp.x,tmp.y,tmp.z,tmp.w), colorBrickPool, newCoords.x*sizeof(uchar4), newCoords.y, newCoords.z);

    // left edge
    tmp = make_float4(0,0,0,0);
    tmp.x += static_cast<float>(colors[4].x);
    tmp.y += static_cast<float>(colors[4].y);
    tmp.z += static_cast<float>(colors[4].z);
    tmp.w += static_cast<float>(colors[4].w);

    tmp.x += static_cast<float>(colors[6].x);
    tmp.y += static_cast<float>(colors[6].y);
    tmp.z += static_cast<float>(colors[6].z);
    tmp.w += static_cast<float>(colors[6].w);

    tmp.x *= 0.5f;
    tmp.y *= 0.5f;
    tmp.z *= 0.5f;
    tmp.w *= 0.5f;

    newCoords = make_uint3(0,1,2);
    newCoords.x+=brickCoords.x;
    newCoords.y+=brickCoords.y;
    newCoords.z+=brickCoords.z;

    surf3Dwrite(make_uchar4(tmp.x,tmp.y,tmp.z,tmp.w), colorBrickPool, newCoords.x*sizeof(uchar4), newCoords.y, newCoords.z);

    // right edge
    tmp = make_float4(0,0,0,0);
    tmp.x += static_cast<float>(colors[5].x);
    tmp.y += static_cast<float>(colors[5].y);
    tmp.z += static_cast<float>(colors[5].z);
    tmp.w += static_cast<float>(colors[5].w);

    tmp.x += static_cast<float>(colors[7].x);
    tmp.y += static_cast<float>(colors[7].y);
    tmp.z += static_cast<float>(colors[7].z);
    tmp.w += static_cast<float>(colors[7].w);

    tmp.x *= 0.5f;
    tmp.y *= 0.5f;
    tmp.z *= 0.5f;
    tmp.w *= 0.5f;

    newCoords = make_uint3(2,1,2);
    newCoords.x+=brickCoords.x;
    newCoords.y+=brickCoords.y;
    newCoords.z+=brickCoords.z;

    surf3Dwrite(make_uchar4(tmp.x,tmp.y,tmp.z,tmp.w), colorBrickPool, newCoords.x*sizeof(uchar4), newCoords.y, newCoords.z);


	// ####################### EDGES (LEFT) #####################

	//bottom edge

	tmp = make_float4(0, 0, 0, 0);
	tmp.x += static_cast<float>(colors[2].x);
	tmp.y += static_cast<float>(colors[2].y);
	tmp.z += static_cast<float>(colors[2].z);
	tmp.w += static_cast<float>(colors[2].w);

	tmp.x += static_cast<float>(colors[6].x);
	tmp.y += static_cast<float>(colors[6].y);
	tmp.z += static_cast<float>(colors[6].z);
	tmp.w += static_cast<float>(colors[6].w);

	tmp.x *= 0.5f;
	tmp.y *= 0.5f;
	tmp.z *= 0.5f;
	tmp.w *= 0.5f;

	newCoords = make_uint3(0,2,1);
	newCoords.x += brickCoords.x;
	newCoords.y += brickCoords.y;
	newCoords.z += brickCoords.z;

	surf3Dwrite(make_uchar4(tmp.x, tmp.y, tmp.z, tmp.w), colorBrickPool, newCoords.x*sizeof(uchar4), newCoords.y, newCoords.z);

	//top edge

	tmp = make_float4(0, 0, 0, 0);
	tmp.x += static_cast<float>(colors[0].x);
	tmp.y += static_cast<float>(colors[0].y);
	tmp.z += static_cast<float>(colors[0].z);
	tmp.w += static_cast<float>(colors[0].w);

	tmp.x += static_cast<float>(colors[4].x);
	tmp.y += static_cast<float>(colors[4].y);
	tmp.z += static_cast<float>(colors[4].z);
	tmp.w += static_cast<float>(colors[4].w);

	tmp.x *= 0.5f;
	tmp.y *= 0.5f;
	tmp.z *= 0.5f;
	tmp.w *= 0.5f;

	newCoords = make_uint3(0, 0, 1);
	newCoords.x += brickCoords.x;
	newCoords.y += brickCoords.y;
	newCoords.z += brickCoords.z;

	surf3Dwrite(make_uchar4(tmp.x, tmp.y, tmp.z, tmp.w), colorBrickPool, newCoords.x*sizeof(uchar4), newCoords.y, newCoords.z);
	// ####################### EDGES (RIGHT) #####################

	//bottom edge

	tmp = make_float4(0, 0, 0, 0);
	tmp.x += static_cast<float>(colors[3].x);
	tmp.y += static_cast<float>(colors[3].y);
	tmp.z += static_cast<float>(colors[3].z);
	tmp.w += static_cast<float>(colors[3].w);

	tmp.x += static_cast<float>(colors[7].x);
	tmp.y += static_cast<float>(colors[7].y);
	tmp.z += static_cast<float>(colors[7].z);
	tmp.w += static_cast<float>(colors[7].w);

	tmp.x *= 0.5f;
	tmp.y *= 0.5f;
	tmp.z *= 0.5f;
	tmp.w *= 0.5f;

	newCoords = make_uint3(2, 2, 1);
	newCoords.x += brickCoords.x;
	newCoords.y += brickCoords.y;
	newCoords.z += brickCoords.z;

	surf3Dwrite(make_uchar4(tmp.x, tmp.y, tmp.z, tmp.w), colorBrickPool, newCoords.x*sizeof(uchar4), newCoords.y, newCoords.z);

	//top edge

	tmp = make_float4(0, 0, 0, 0);
	tmp.x += static_cast<float>(colors[1].x);
	tmp.y += static_cast<float>(colors[1].y);
	tmp.z += static_cast<float>(colors[1].z);
	tmp.w += static_cast<float>(colors[1].w);

	tmp.x += static_cast<float>(colors[5].x);
	tmp.y += static_cast<float>(colors[5].y);
	tmp.z += static_cast<float>(colors[5].z);
	tmp.w += static_cast<float>(colors[5].w);

	tmp.x *= 0.5f;
	tmp.y *= 0.5f;
	tmp.z *= 0.5f;
	tmp.w *= 0.5f;

	newCoords = make_uint3(2, 0, 1);
	newCoords.x += brickCoords.x;
	newCoords.y += brickCoords.y;
	newCoords.z += brickCoords.z;

	surf3Dwrite(make_uchar4(tmp.x, tmp.y, tmp.z, tmp.w), colorBrickPool, newCoords.x*sizeof(uchar4), newCoords.y, newCoords.z);


}


#endif