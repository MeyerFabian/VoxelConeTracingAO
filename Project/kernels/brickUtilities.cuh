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

__device__ unsigned int addColors(const uchar4 &input, float4 &output)
{
    if(input.w != 0)
    {
        output.x += input.x;
        output.y += input.y;
        output.z += input.z;
        output.w += input.w;
        return 1;
    }
    else
        return 0;
}

// filters the brick with an inverse gaussian mask to fill a brick
// the corner bricks are used as sources
// 19 Voxels to be filled
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
	
    // ################ center: ####################### 1. checked
    float4 tmp = make_float4(0,0,0,0);
    int counter = 0;


    counter += addColors(colors[0], tmp);
    counter += addColors(colors[1], tmp);
    counter += addColors(colors[2], tmp);
    counter += addColors(colors[3], tmp);
    counter += addColors(colors[4], tmp);
    counter += addColors(colors[5], tmp);
    counter += addColors(colors[6], tmp);
    counter += addColors(colors[7], tmp);

    tmp.x /= counter;
    tmp.y /= counter;
    tmp.z /= counter;
    tmp.w /= counter;

    counter = 0;

    uint3 newCoords = make_uint3(1,1,1);
    newCoords.x+=brickCoords.x;
    newCoords.y+=brickCoords.y;
    newCoords.z+=brickCoords.z;

    __syncthreads();

    surf3Dwrite(make_uchar4(tmp.x,tmp.y,tmp.z,tmp.w), colorBrickPool, newCoords.x*sizeof(uchar4), newCoords.y, newCoords.z);

    // ################### FACES ########################## 2. checked
    // right side: 1, 3, 5, 7

    /*
     * // front corners
    insertpos[0] = make_uint3(0,0,0);
    insertpos[1] = make_uint3(2,0,0);
    insertpos[2] = make_uint3(0,2,0);
    insertpos[3] = make_uint3(2,2,0);

    //back corners
    insertpos[4] = make_uint3(0,0,2);
    insertpos[5] = make_uint3(2,0,2);
    insertpos[6] = make_uint3(0,2,2);
    insertpos[7] = make_uint3(2,2,2);
     */

    tmp = make_float4(0,0,0,0);
    counter += addColors(colors[1], tmp);
    counter += addColors(colors[3], tmp);
    counter += addColors(colors[5], tmp);
    counter += addColors(colors[7], tmp);

    tmp.x /= counter;
    tmp.y /= counter;
    tmp.z /= counter;
    tmp.w /= counter;

    counter = 0;

    newCoords = make_uint3(2,1,1);
    newCoords.x+=brickCoords.x;
    newCoords.y+=brickCoords.y;
    newCoords.z+=brickCoords.z;

    surf3Dwrite(make_uchar4(tmp.x,tmp.y,tmp.z,tmp.w), colorBrickPool, newCoords.x*sizeof(uchar4), newCoords.y, newCoords.z);

    // left side: 0, 2, 4, 6 ### 3. error corrected
    tmp = make_float4(0,0,0,0);
    counter += addColors(colors[0], tmp);
    counter += addColors(colors[2], tmp);
    counter += addColors(colors[4], tmp);
    counter += addColors(colors[6], tmp);

    tmp.x /= counter;
    tmp.y /= counter;
    tmp.z /= counter;
    tmp.w /= counter;

    counter = 0;

    newCoords = make_uint3(0,1,1);
    newCoords.x+=brickCoords.x;
    newCoords.y+=brickCoords.y;
    newCoords.z+=brickCoords.z;

    surf3Dwrite(make_uchar4(tmp.x,tmp.y,tmp.z,tmp.w), colorBrickPool, newCoords.x*sizeof(uchar4), newCoords.y, newCoords.z);

    // bottom side: 2, 3, 6, 7 ### 4. checked
    tmp = make_float4(0,0,0,0);

    counter += addColors(colors[2], tmp);
    counter += addColors(colors[3], tmp);
    counter += addColors(colors[6], tmp);
    counter += addColors(colors[7], tmp);

    tmp.x /= counter;
    tmp.y /= counter;
    tmp.z /= counter;
    tmp.w /= counter;

    counter = 0;

    newCoords = make_uint3(1,2,1);
    newCoords.x+=brickCoords.x;
    newCoords.y+=brickCoords.y;
    newCoords.z+=brickCoords.z;

    surf3Dwrite(make_uchar4(tmp.x,tmp.y,tmp.z,tmp.w), colorBrickPool, newCoords.x*sizeof(uchar4), newCoords.y, newCoords.z);

    // top side: 0, 1, 4, 5 ### 5. checked
    tmp = make_float4(0,0,0,0);

    counter += addColors(colors[0], tmp);
    counter += addColors(colors[1], tmp);
    counter += addColors(colors[4], tmp);
    counter += addColors(colors[5], tmp);

    tmp.x /= counter;
    tmp.y /= counter;
    tmp.z /= counter;
    tmp.w /= counter;

    counter = 0;

    newCoords = make_uint3(1,0,1);
    newCoords.x+=brickCoords.x;
    newCoords.y+=brickCoords.y;
    newCoords.z+=brickCoords.z;

    surf3Dwrite(make_uchar4(tmp.x,tmp.y,tmp.z,tmp.w), colorBrickPool, newCoords.x*sizeof(uchar4), newCoords.y, newCoords.z);

    // near side: 0, 1, 2, 3 ### 6. checked
    tmp = make_float4(0,0,0,0);

    counter += addColors(colors[0], tmp);
    counter += addColors(colors[1], tmp);
    counter += addColors(colors[2], tmp);
    counter += addColors(colors[3], tmp);

    tmp.x /= counter;
    tmp.y /= counter;
    tmp.z /= counter;
    tmp.w /= counter;

    counter = 0;

    newCoords = make_uint3(1,1,0);
    newCoords.x+=brickCoords.x;
    newCoords.y+=brickCoords.y;
    newCoords.z+=brickCoords.z;

    surf3Dwrite(make_uchar4(tmp.x,tmp.y,tmp.z,tmp.w), colorBrickPool, newCoords.x*sizeof(uchar4), newCoords.y, newCoords.z);

    // far side: 4, 5, 6, 7 ### 7. checked
    tmp = make_float4(0,0,0,0);

    counter += addColors(colors[4], tmp);
    counter += addColors(colors[5], tmp);
    counter += addColors(colors[6], tmp);
    counter += addColors(colors[7], tmp);

    tmp.x /= counter;
    tmp.y /= counter;
    tmp.z /= counter;
    tmp.w /= counter;

    counter = 0;

    newCoords = make_uint3(1,1,2);
    newCoords.x+=brickCoords.x;
    newCoords.y+=brickCoords.y;
    newCoords.z+=brickCoords.z;

    surf3Dwrite(make_uchar4(tmp.x,tmp.y,tmp.z,tmp.w), colorBrickPool, newCoords.x*sizeof(uchar4), newCoords.y, newCoords.z);

    // ####################### EDGES (FRONT) #####################
    // top edge: 0, 1 ### 8. checked
    tmp = make_float4(0,0,0,0);

    counter += addColors(colors[0], tmp);
    counter += addColors(colors[1], tmp);

    tmp.x /= counter;
    tmp.y /= counter;
    tmp.z /= counter;
    tmp.w /= counter;

    counter = 0;

    newCoords = make_uint3(1,0,0);
    newCoords.x+=brickCoords.x;
    newCoords.y+=brickCoords.y;
    newCoords.z+=brickCoords.z;

    surf3Dwrite(make_uchar4(tmp.x,tmp.y,tmp.z,tmp.w), colorBrickPool, newCoords.x*sizeof(uchar4), newCoords.y, newCoords.z);

    // bottom edge: 2, 3 ### 9. checked
    tmp = make_float4(0,0,0,0);

    counter += addColors(colors[2], tmp);
    counter += addColors(colors[3], tmp);

    tmp.x /= counter;
    tmp.y /= counter;
    tmp.z /= counter;
    tmp.w /= counter;

    counter = 0;

    newCoords = make_uint3(1,2,0);
    newCoords.x+=brickCoords.x;
    newCoords.y+=brickCoords.y;
    newCoords.z+=brickCoords.z;

    surf3Dwrite(make_uchar4(tmp.x,tmp.y,tmp.z,tmp.w), colorBrickPool, newCoords.x*sizeof(uchar4), newCoords.y, newCoords.z);

    // left edge: 0, 2 ### 10. checked
    tmp = make_float4(0,0,0,0);

    counter += addColors(colors[0], tmp);
    counter += addColors(colors[2], tmp);

    tmp.x /= counter;
    tmp.y /= counter;
    tmp.z /= counter;
    tmp.w /= counter;

    counter = 0;

    newCoords = make_uint3(0,1,0);
    newCoords.x+=brickCoords.x;
    newCoords.y+=brickCoords.y;
    newCoords.z+=brickCoords.z;

    surf3Dwrite(make_uchar4(tmp.x,tmp.y,tmp.z,tmp.w), colorBrickPool, newCoords.x*sizeof(uchar4), newCoords.y, newCoords.z);

    // right edge: 1,3 ### 11. checked
    tmp = make_float4(0,0,0,0);

    counter += addColors(colors[1], tmp);
    counter += addColors(colors[3], tmp);

    tmp.x /= counter;
    tmp.y /= counter;
    tmp.z /= counter;
    tmp.w /= counter;

    counter = 0;

    newCoords = make_uint3(2,1,0);
    newCoords.x+=brickCoords.x;
    newCoords.y+=brickCoords.y;
    newCoords.z+=brickCoords.z;

    surf3Dwrite(make_uchar4(tmp.x,tmp.y,tmp.z,tmp.w), colorBrickPool, newCoords.x*sizeof(uchar4), newCoords.y, newCoords.z);

    // ####################### EDGES (BACK) #####################
    // top edge: 4,5 ### 12. checked
    tmp = make_float4(0,0,0,0);
    counter += addColors(colors[4], tmp);
    counter += addColors(colors[5], tmp);

    tmp.x /= counter;
    tmp.y /= counter;
    tmp.z /= counter;
    tmp.w /= counter;

    counter = 0;

    newCoords = make_uint3(1,0,2);
    newCoords.x+=brickCoords.x;
    newCoords.y+=brickCoords.y;
    newCoords.z+=brickCoords.z;

    surf3Dwrite(make_uchar4(tmp.x,tmp.y,tmp.z,tmp.w), colorBrickPool, newCoords.x*sizeof(uchar4), newCoords.y, newCoords.z);

    // bottom edge: 6,7 ### 13. error corrected
    tmp = make_float4(0,0,0,0);

    counter += addColors(colors[6], tmp);
    counter += addColors(colors[7], tmp);

    tmp.x /= counter;
    tmp.y /= counter;
    tmp.z /= counter;
    tmp.w /= counter;

    counter = 0;

    newCoords = make_uint3(1,2,2);
    newCoords.x+=brickCoords.x;
    newCoords.y+=brickCoords.y;
    newCoords.z+=brickCoords.z;

    surf3Dwrite(make_uchar4(tmp.x,tmp.y,tmp.z,tmp.w), colorBrickPool, newCoords.x*sizeof(uchar4), newCoords.y, newCoords.z);

    // left edge: 4,6 ### 14. checked
    tmp = make_float4(0,0,0,0);

    counter += addColors(colors[4], tmp);
    counter += addColors(colors[6], tmp);

    tmp.x /= counter;
    tmp.y /= counter;
    tmp.z /= counter;
    tmp.w /= counter;

    counter = 0;

    newCoords = make_uint3(0,1,2);
    newCoords.x+=brickCoords.x;
    newCoords.y+=brickCoords.y;
    newCoords.z+=brickCoords.z;

    surf3Dwrite(make_uchar4(tmp.x,tmp.y,tmp.z,tmp.w), colorBrickPool, newCoords.x*sizeof(uchar4), newCoords.y, newCoords.z);

    // right edge: 5,7 ### 15. checked 
    tmp = make_float4(0,0,0,0);

    counter += addColors(colors[5], tmp);
    counter += addColors(colors[7], tmp);

    tmp.x /= counter;
    tmp.y /= counter;
    tmp.z /= counter;
    tmp.w /= counter;

    counter = 0;

    newCoords = make_uint3(2,1,2);
    newCoords.x+=brickCoords.x;
    newCoords.y+=brickCoords.y;
    newCoords.z+=brickCoords.z;

    surf3Dwrite(make_uchar4(tmp.x,tmp.y,tmp.z,tmp.w), colorBrickPool, newCoords.x*sizeof(uchar4), newCoords.y, newCoords.z);


	// ####################### EDGES (LEFT) #####################

	//top edge : 0,4 ### 16. checked

	tmp = make_float4(0, 0, 0, 0);

	counter += addColors(colors[0], tmp);
	counter += addColors(colors[4], tmp);

	tmp.x /= counter;
	tmp.y /= counter;
	tmp.z /= counter;
	tmp.w /= counter;

	counter = 0;

	newCoords = make_uint3(0, 0, 1);
	newCoords.x += brickCoords.x;
	newCoords.y += brickCoords.y;
	newCoords.z += brickCoords.z;

	surf3Dwrite(make_uchar4(tmp.x, tmp.y, tmp.z, tmp.w), colorBrickPool, newCoords.x*sizeof(uchar4), newCoords.y, newCoords.z);

	//bottom edge: 2,6 ### 17.checked 

	tmp = make_float4(0, 0, 0, 0);

    counter += addColors(colors[2], tmp);
    counter += addColors(colors[6], tmp);

    tmp.x /= counter;
    tmp.y /= counter;
    tmp.z /= counter;
    tmp.w /= counter;

    counter = 0;

	newCoords = make_uint3(0,2,1);
	newCoords.x += brickCoords.x;
	newCoords.y += brickCoords.y;
	newCoords.z += brickCoords.z;

	surf3Dwrite(make_uchar4(tmp.x, tmp.y, tmp.z, tmp.w), colorBrickPool, newCoords.x*sizeof(uchar4), newCoords.y, newCoords.z);

	// ####################### EDGES (RIGHT) #####################

	//top edge: 5,1 ### 18. checked

	tmp = make_float4(0, 0, 0, 0);

	counter += addColors(colors[1], tmp);
	counter += addColors(colors[5], tmp);

	tmp.x /= counter;
	tmp.y /= counter;
	tmp.z /= counter;
	tmp.w /= counter;

	counter = 0;

	newCoords = make_uint3(2, 0, 1);
	newCoords.x += brickCoords.x;
	newCoords.y += brickCoords.y;
	newCoords.z += brickCoords.z;

	surf3Dwrite(make_uchar4(tmp.x, tmp.y, tmp.z, tmp.w), colorBrickPool, newCoords.x*sizeof(uchar4), newCoords.y, newCoords.z);

	//bottom edge: 3,7 ### 19.checked

	tmp = make_float4(0, 0, 0, 0);

    counter += addColors(colors[3], tmp);
    counter += addColors(colors[7], tmp);

    tmp.x /= counter;
    tmp.y /= counter;
    tmp.z /= counter;
    tmp.w /= counter;

    counter = 0;

	newCoords = make_uint3(2, 2, 1);
	newCoords.x += brickCoords.x;
	newCoords.y += brickCoords.y;
	newCoords.z += brickCoords.z;

	surf3Dwrite(make_uchar4(tmp.x, tmp.y, tmp.z, tmp.w), colorBrickPool, newCoords.x*sizeof(uchar4), newCoords.y, newCoords.z);

}


#endif