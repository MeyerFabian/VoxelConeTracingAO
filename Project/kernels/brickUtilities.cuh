#ifndef BRICK_UTILITIES_CUH
#define BRICK_UTILITIES_CUH

__device__ uint3 getBrickCoords(unsigned int brickAdress, unsigned int brickPoolSideLength, unsigned int brickSideLength = 3)
{
    uint3 coords;
    brickPoolSideLength /=3;
    coords.x = brickAdress / (brickPoolSideLength*brickPoolSideLength);
    coords.y = (brickAdress / brickPoolSideLength) % brickPoolSideLength;
    coords.z = brickAdress % brickPoolSideLength;

    coords.x = coords.x*brickSideLength;
    coords.y = coords.y*brickSideLength;
    coords.z = coords.z*brickSideLength;

    return coords;
}

__device__ unsigned int encodeBrickCoords(uint3 coords)
{
unsigned int codeX = ((0x000003FF & coords.x) << 20U);
unsigned int codeY = ((0x000003FF & coords.y) << 10U);
unsigned int codeZ = ((0x000003FF & coords.z));
unsigned int code = codeX | codeY | codeZ;

return code;
}

__device__ uint3 decodeBrickCoords(unsigned int coded)
{
    uint3 coords;
    coords.z = coded & 0x000003FF;
    coords.y = (coded & 0x000FFC00) >> 10U;
    coords.x = (coded & 0x3FF00000) >> 20U;
    return coords;
}

#endif