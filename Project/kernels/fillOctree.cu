#include <stdio.h>
#include <cuda_runtime.h>
#include "fillOctree.cuh"


const int maxNodePoolSize = 1024;
int volumeResolution = 384;

bool constantMemoryValid = false;   // the flag indicates wheather a kernel is allowed to use the constantNodePool
__constant__ node constNodePool[maxNodePoolSize];
__constant__ int constVolumeResolution[1];
__device__ unsigned int globalNodePoolCounter = 0;
__device__ unsigned int globalBrickPoolCounter = 0;

surface<void, cudaSurfaceType3D> surfRef;

cudaError_t setVolumeResulution(int resolution)
{
    volumeResolution = resolution;
    cudaError_t errorCode = cudaMemcpyToSymbol(constVolumeResolution, &resolution, sizeof(int));
    return errorCode;
}

__device__
unsigned int getBit(unsigned int value, unsigned int position)
{
    return (value >> (position-1)) & 1u;
}

__device__
void setBit(unsigned int &value, unsigned int position)
{
    value |= (1u << (position-1));
}

__device__
void unSetBit(unsigned int &value, unsigned int position)
{
    value &= ~(1u << (position-1));
}

__global__
void clearNodePoolKernel(node *nodePool, int poolSize)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if(i >= poolSize)
        return;

    nodePool[i].nodeTilePointer = 0;
    nodePool[i].value = 0;
}

__global__
void clearCounter()
{
    globalNodePoolCounter = 0;
    globalBrickPoolCounter = 0;
}

__device__ uint3 getBrickCoords(unsigned int brickAdress, unsigned int brickPoolSideLength, unsigned int brickSideLength = 3)
{
    uint3 coords;
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
    return (0x000003FF & coords.x) << 20U | (0x000003FF & coords.y) << 10U | (0x000003FF & coords.z);
}

__device__ uint3 decodeBrickCoords(unsigned int coded)
{
    uint3 coords;
    coords.z = coded & 0x000003FF;
    coords.y = (coded >> 10) & 0x000003FF;
    coords.x = (coded >> 20) & 0x000003FF;
    return coords;
}

__device__ void filterBrick(const uint3 &brickCoords)
{
    // TODO: filter brick
}

__global__ void filterBrickCorners(node *nodePool, int maxNodes, int maxLevel)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if(index >= maxNodes)
        return;

    unsigned int nodeOffset = 0;
    unsigned int childPointer = 0;
    unsigned int offset = 0;

    uint3 octants[8];
    octants[0] = make_uint3(0,0,0);
    octants[1] = make_uint3(0,0,1);
    octants[2] = make_uint3(0,1,0);
    octants[3] = make_uint3(0,1,1);
    octants[4] = make_uint3(1,0,0);
    octants[5] = make_uint3(1,0,1);
    octants[6] = make_uint3(1,1,0);
    octants[7] = make_uint3(1,1,1);

    uint3 nextOctant;
    unsigned int octantIdx = 0;

    for (int i = 0; i <=maxLevel; i++)
    {
        if(i==0)
            octantIdx = 0;
        else
            octantIdx = (index / static_cast<unsigned int>(pow(8.f, static_cast<float>(i-1)))) % 8;

        nextOctant = octants[octantIdx];

        // make the octant position 1D for the linear memory
        nodeOffset = nextOctant.x + 2 * nextOctant.y + 4 * nextOctant.z;
        offset = nodeOffset + childPointer * 8;

        unsigned int pointer = nodePool[offset].nodeTilePointer;

        // traverse further until we reach a non valid point. as we wish to iterate to the bottom, we return if there is an inalid connectioon (should not happen)
        if(getBit(pointer, 32) == 1)
            childPointer = pointer & 0x3fffffff;
        else if(i == maxLevel)
        {
            unsigned int value = nodePool[offset].value;
            if (getBit(value, 32) == 1)
                filterBrick(decodeBrickCoords(value & 0x3fffffff));
        }
    }
}

__device__ void fillBrickCorners(const uint3 &brickCoords, const float3 &voxelPosition, const uchar4 &color)
{
    uint3 nextOctant;
    nextOctant.x = static_cast<unsigned int>(2 * voxelPosition.x);
    nextOctant.y = static_cast<unsigned int>(2 * voxelPosition.y);
    nextOctant.z = static_cast<unsigned int>(2 * voxelPosition.z);

    unsigned int offset = nextOctant.x + 2 * nextOctant.y + 4 * nextOctant.z;

    // here we have our possible brick corners // TODO: fill them in const memory maybe?
    uint3 insertPositions[8];
    // front corners
    insertPositions[0] = make_uint3(0,0,0);
    insertPositions[1] = make_uint3(2,0,0);
    insertPositions[2] = make_uint3(2,2,0);
    insertPositions[3] = make_uint3(0,2,0);

    //back corners
    insertPositions[4] = make_uint3(0,0,2);
    insertPositions[5] = make_uint3(2,0,2);
    insertPositions[6] = make_uint3(2,2,2);
    insertPositions[7] = make_uint3(0,2,2);

    /*
    if(brickCoords.x == 0 && brickCoords.y == 609 && brickCoords.z == 840) {
        printf("offset : %d\n", offset);
        printf("color r: %d g: %d b: %d\n", static_cast<unsigned int>(color.x), color.y, color.z);
    }*/

    uint3 pos = insertPositions[offset];

    // write the color value to the corner TODO: use a shared counter to prevent race conditions between double list entries in the fragment list
    surf3Dwrite(color, surfRef, pos.x*sizeof(uchar4), pos.y, pos.z);
}

__global__ void insertVoxelsInLastLevel(node *nodePool, uint1 *positionBuffer, uchar4* colorBufferDevPointer, unsigned int maxLevel, int fragmentListSize)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if(index >= fragmentListSize)
        return;

    const unsigned int mask_bits = 0x000003FF;
    unsigned int codedPosition = positionBuffer[index].x;

    float3 position;
    // dont forget the .f for casting reasons :P
    position.x = ((codedPosition) & (mask_bits)) / 1024.f;
    position.y = ((codedPosition >> 10) & (mask_bits)) / 1024.f;
    position.z = ((codedPosition >> 20) & (mask_bits)) / 1024.f;

    unsigned int nodeOffset = 0;
    unsigned int childPointer = 0;
    unsigned int offset=0;
    unsigned int nodeTile = 0;
    unsigned int value = 0;

    for (int i = 0; i < maxLevel; i++)
    {
        uint3 nextOctant = make_uint3(0, 0, 0);
        // determine octant for the given voxel
        if(i != 0)
        {
            nextOctant.x = static_cast<unsigned int>(2 * position.x);
            nextOctant.y = static_cast<unsigned int>(2 * position.y);
            nextOctant.z = static_cast<unsigned int>(2 * position.z);
        }

        // make the octant position 1D for the linear memory
        nodeOffset = nextOctant.x + 2 * nextOctant.y + 4 * nextOctant.z;
        offset = nodeOffset + childPointer * 8;

        nodeTile = nodePool[offset].nodeTilePointer;

        childPointer = nodeTile & 0x3fffffff;

        if(i != 0)
        {
            position.x = 2 * position.x - nextOctant.x;
            position.y = 2 * position.y - nextOctant.y;
            position.z = 2 * position.z - nextOctant.z;
        }
    }

    // now we fill the corners of our bricks at the last level. This level is represented with 8 values inside a brick
    value = nodePool[offset].value;

    if(getBit(value,32) == 1)
    {
        // we have a valid brick => fill it
        fillBrickCorners(decodeBrickCoords(value), position, colorBufferDevPointer[index]);
    }
}

__global__ void markNodeForSubdivision(node *nodePool, int poolSize, int maxLevel, uint1* positionBuffer, int fragmentListSize)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if(index >= fragmentListSize)
        return;

    // mask to get 10 bit position coords
    const unsigned int mask_bits = 0x000003FF;
    unsigned int codedPosition = positionBuffer[index].x;

    float3 position;
    // dont forget the .f for casting reasons :P
    position.x = ((codedPosition) & (mask_bits)) / 1024.f;
    position.y = ((codedPosition >> 10) & (mask_bits)) / 1024.f;
    position.z = ((codedPosition >> 20) & (mask_bits)) / 1024.f;


    unsigned int nodeOffset = 0;
    unsigned int childPointer = 0;

    for(int i=0;i<=maxLevel;i++)
    {
        uint3 nextOctant = make_uint3(0, 0, 0);
        // determine octant for the given voxel
        unsigned int offset = 0;
        if(i != 0)
        {
            nextOctant.x = static_cast<unsigned int>(2 * position.x);
            nextOctant.y = static_cast<unsigned int>(2 * position.y);
            nextOctant.z = static_cast<unsigned int>(2 * position.z);

            // make the octant position 1D for the linear memory
            nodeOffset = nextOctant.x + 2 * nextOctant.y + 4 * nextOctant.z;
            offset = nodeOffset + childPointer * 8;
        }

        // the maxdivide bit indicates wheather the node has children 1 means has children 0 means does not have children
        unsigned int nodeTile = nodePool[offset].nodeTilePointer;
        __syncthreads();
        unsigned int maxDivide = getBit(nodeTile,32);

        if(maxDivide == 0)
        {
            // as the node has no children we set the second bit to 1 which indicates that memory should be allocated
            setBit(nodeTile,31); // possible race condition but it is not importatnt in our case
            nodePool[offset].nodeTilePointer = nodeTile;
            __syncthreads();
            break;
        }
        else
        {
            // if the node has children we read the pointer to the next nodetile
            childPointer = nodeTile & 0x3fffffff;
        }

        if(i!=0)
        {
            position.x = 2 * position.x - nextOctant.x;
            position.y = 2 * position.y - nextOctant.y;
            position.z = 2 * position.z - nextOctant.z;
        }
    }
}

__global__ void reserveMemoryForNodes(node *nodePool, int maxNodes, int level, unsigned int* counter, unsigned int brickPoolResolution, unsigned int brickResolution, int lastLevel)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if(index >= maxNodes)
        return;

    unsigned int nodeOffset = 0;
    unsigned int childPointer = 0;

    uint3 octants[8];
    octants[0] = make_uint3(0,0,0);
    octants[1] = make_uint3(0,0,1);
    octants[2] = make_uint3(0,1,0);
    octants[3] = make_uint3(0,1,1);
    octants[4] = make_uint3(1,0,0);
    octants[5] = make_uint3(1,0,1);
    octants[6] = make_uint3(1,1,0);
    octants[7] = make_uint3(1,1,1);

    uint3 nextOctant;
    unsigned int octantIdx = 0;

    for (int i = 0; i <=level; i++)
    {
        if(i==0)
            octantIdx = 0;
        else
            octantIdx = (index / static_cast<unsigned int>(pow(8.f, static_cast<float>(i-1)))) % 8;

        nextOctant = octants[octantIdx];

        // make the octant position 1D for the linear memory
        nodeOffset = nextOctant.x + 2 * nextOctant.y + 4 * nextOctant.z;

        unsigned int offset = nodeOffset + childPointer * 8;

        unsigned int pointer = nodePool[offset].nodeTilePointer;
        unsigned int value = nodePool[offset].value;
        __syncthreads();    //make sure all threads have a valid nodeTilePointer

        unsigned int reserve = getBit(pointer, 31);
        unsigned int maxDivided = getBit(pointer, 32);
        if (reserve == 1)
        {
            // increment the global nodecount and allocate the memory in our
            unsigned int adress = atomicAdd(counter, 1) + 1;
            unsigned int brickAdress = atomicAdd(&globalBrickPoolCounter, 1);

            pointer = (adress & 0x3fffffff) | pointer;
            value = encodeBrickCoords(getBrickCoords(brickAdress, brickPoolResolution, brickResolution));

            // set the first bit to 1. this indicates, that we use the texture brick instead of a constant value as color.
            setBit(value, 32);
            setBit(pointer, 32);

            // make sure we don't reserve the same nodeTile next time :)
            unSetBit(pointer, 31);

            if(lastLevel == 1)
                unSetBit(pointer,32);

            nodePool[offset].nodeTilePointer = pointer;
            nodePool[offset].value = value;

            __syncthreads();
            break;
        }
        else
        {
            // traverse further
            childPointer = pointer & 0x3fffffff;
        }
    }

}

cudaError_t buildSVO(node *nodePool,
                     int poolSize,
                     cudaArray_t *brickPool,
                     dim3 textureDim,
                     uint1* positionDevPointer,
                     uchar4* colorBufferDevPointer,
                     uchar4* normalDevPointer,
                     int fragmentListSize)
{
    cudaError_t errorCode = cudaSuccess;
    // calculate maxlevel
    int maxLevel = static_cast<int>(log((volumeResolution*volumeResolution*volumeResolution))/log(8));
    // note that we dont calculate +1 as we store 8 voxels per brick

    dim3 block_dim(32, 0, 0);
    dim3 grid_dim(fragmentListSize/block_dim.x, 0, 0);

    int threadsPerBlock = 64;
    int blockCount = fragmentListSize / threadsPerBlock;


    unsigned int *h_counter = new unsigned int[1];
    unsigned int *d_counter;
    *h_counter = 0;

    cudaMalloc(&d_counter, sizeof(int));
    cudaMemcpy(d_counter,h_counter,sizeof(unsigned int),cudaMemcpyHostToDevice);

    clearCounter<<<1,1>>>();
    cudaDeviceSynchronize();

    int lastLevel = 0;

    for(int i=0;i<maxLevel;i++)
    {
        markNodeForSubdivision<<<blockCount, threadsPerBlock>>>(nodePool, poolSize, i, positionDevPointer, fragmentListSize);
        cudaDeviceSynchronize();
        unsigned int maxNodes = static_cast<unsigned int>(pow(8,i));

        const unsigned int threadPerBlockReserve = 512;
        int blockCountReserve = maxNodes;

        if(maxNodes >= threadPerBlockReserve)
            blockCountReserve = maxNodes / threadPerBlockReserve;

        if(i == maxLevel-1)
            lastLevel = 1;

        reserveMemoryForNodes <<< blockCountReserve, threadPerBlockReserve >>> (nodePool, maxNodes, i, d_counter, volumeResolution, 3, lastLevel);
        cudaDeviceSynchronize();

        /*
        cudaMemcpy(h_counter, d_counter, sizeof(unsigned int), cudaMemcpyDeviceToHost);
        printf("reserved node tiles: %d\n", *h_counter);*/
    }

    cudaDeviceSynchronize();
    insertVoxelsInLastLevel<<<blockCount,threadsPerBlock>>>(nodePool,positionDevPointer,colorBufferDevPointer,maxLevel, fragmentListSize);

    const unsigned int threadPerBlockSpread = 512;
    unsigned int blockCountSpread;
    unsigned int nodeCount = static_cast<unsigned int>(pow(8,maxLevel-1));

    blockCountSpread = nodeCount;

    if(nodeCount >= threadPerBlockSpread)
        blockCountSpread = nodeCount / threadPerBlockSpread;

    cudaDeviceSynchronize();
    filterBrickCorners<<<blockCountSpread, threadPerBlockSpread>>>(nodePool, nodeCount, maxLevel);

    cudaFree(d_counter);
    delete h_counter;

    return errorCode;
}

cudaError_t clearNodePoolCuda(node *nodePool, int poolSize)
{
    cudaError_t errorCode = cudaSuccess;
    int threadsPerBlock = 64;
    int blockCount = poolSize / threadsPerBlock;

    clearNodePoolKernel<<<blockCount, threadsPerBlock>>>(nodePool, poolSize);

    return errorCode;
}