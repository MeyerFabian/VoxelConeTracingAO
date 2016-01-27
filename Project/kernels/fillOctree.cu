#include <stdio.h>
#include <cuda_runtime.h>

#include "globalResources.cuh"
#include "fillOctree.cuh"
#include "bitUtilities.cuh"
#include "octreeMipMapping.cuh"
#include "brickUtilities.cuh"


cudaError_t setVolumeResulution(int resolution)
{
    volumeResolution = resolution;
    cudaError_t errorCode = cudaMemcpyToSymbol(constVolumeResolution, &resolution, sizeof(int));
    return errorCode;
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

// resets the global counters
__global__
void clearCounter()
{
    globalNodePoolCounter = 0;
    globalBrickPoolCounter = 0;
}

// traverses to the bottom level and filters all bricks by applying a inverse gaussian mask to the corner voxels
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

    for (int i = 0; i < maxLevel; i++)
    {
        if(i==0)
            octantIdx = 0;
        else // here we make sure that we are able to reach every node within the tree
            octantIdx = (index / static_cast<unsigned int>(pow(8.f, static_cast<float>(i-1)))) % 8;

        nextOctant = octants[octantIdx];

        // make the octant position 1D for the linear memory
        nodeOffset = nextOctant.x + 2 * nextOctant.y + 4 * nextOctant.z;
        offset = nodeOffset + childPointer * 8;

        unsigned int pointer = nodePool[offset].nodeTilePointer;

        // traverse further until we reach a non valid point. as we wish to iterate to the bottom, we return if there is an inalid connectioon (should not happen)
        if(getBit(pointer, 32) == 1)
            childPointer = pointer & 0x3fffffff;
        else if(i == maxLevel-1)
        {
            unsigned int value = nodePool[offset].value;
            if (getBit(value, 32) == 1)// only filter if the brick is used
            {
                filterBrick(decodeBrickCoords(value & 0x3fffffff));
            }
        }
    }
}

// traverses to the bottom level and fills the 8 corners of each brick
// note that the bricks at the bottom level represent an octree level by themselves
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

    for (int i = 0; i <= maxLevel; i++)
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
        __syncthreads();
        if(getBit(nodeTile,32) == 1) {
            childPointer = nodeTile & 0x3fffffff;
        }

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

// traverses the octree with one thread for each entry in the fragmentlist. Marks every node on its way as dividable
// this kernel gets executed successively for each level of the tree
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

// the kernel is launched with a threadCount corresponding to the maximum possible nodecount for the current level
// this kernel gets executed successively for each level of the tree
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
            uint3 test = decodeBrickCoords(value);
            //printf("decode: x:%d, y:%d, z:%d\n" ,test.x, test.y, test.z);

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


    cudaChannelFormatDesc channelDesc;
    errorCode = cudaGetChannelDesc(&channelDesc, *brickPool);
    errorCode = cudaBindSurfaceToArray(&colorBrickPool, *brickPool, &channelDesc);

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