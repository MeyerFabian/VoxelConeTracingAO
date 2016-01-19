#include <stdio.h>
#include <cuda_runtime.h>
#include <src/Utilities/errorUtils.h>
#include "fillOctree.cuh"


const int maxNodePoolSize = 1024;
int volumeResolution = 384;

bool constantMemoryValid = false;   // the flag indicates wheather a kernel is allowed to use the constantNodePool
__constant__ node constNodePool[maxNodePoolSize];
__constant__ int constVolumeResolution[1];
__device__ unsigned int globalNodePoolCounter = 0;

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
void testFilling(dim3 texture_dim)
{
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;
    int z = blockIdx.z*blockDim.z + threadIdx.z;

    if(x >= texture_dim.x || y >= texture_dim.y || z >= texture_dim.z)
    {
        return;
    }

    uchar4 element = make_uchar4(255, 255, 255, 255);
    surf3Dwrite(element, surfRef, x*sizeof(uchar4), y, z);
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
}

__global__ void markNodeForSubdivision(node *nodePool, int poolSize, int maxLevel, uint1* positionBuffer, int volumeSideLength)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;

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
        nextOctant.x = static_cast<unsigned int>(2 * position.x);
        nextOctant.y = static_cast<unsigned int>(2 * position.y);
        nextOctant.z = static_cast<unsigned int>(2 * position.z);

        // make the octant position 1D for the linear memory
        nodeOffset = nextOctant.x + 2*nextOctant.y + 4*nextOctant.z;

        // the maxdivide bit indicates wheather the node has children 1 means has children 0 means does not have children
        unsigned int nodeTile = nodePool[nodeOffset+childPointer*8].nodeTilePointer;
        __syncthreads();
        unsigned int maxDivide = getBit(nodeTile,32);

        if(maxDivide == 0)
        {
            // as the node has no children we set the second bit to 1 which indicates that memory should be allocated
            setBit(nodeTile,31); // possible race condition but it is not importatnt in our case
            nodePool[nodeOffset+childPointer*8].nodeTilePointer = nodeTile;
            __syncthreads();
            break;
        }
        else
        {
            // if the node has children we read the pointer to the next nodetile
            childPointer = nodeTile & 0x3fffffff;
        }

        position.x = 2*position.x - nextOctant.x;
        position.y = 2*position.y - nextOctant.y;
        position.z = 2*position.z - nextOctant.z;
    }
}

__global__ void reserveMemoryForNodes(node *nodePool, int maxNodes, int level, unsigned int* counter)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if(index >= maxNodes)
        return;

    /*
    double3 position;
    // make sure we traverse all nodes => the position is between 0 and 1
    unsigned int sideLength = static_cast<unsigned int>(cbrtf(powf(8,level)));


    position.x = (index / sideLength*sideLength)/sideLength;
    position.y = ((index / sideLength) % sideLength)/sideLength;
    position.z = (index % sideLength) / sideLength;
     */

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

    for(int i=0;i<=level;i++)
    {
        uint3 nextOctant = octants[index/static_cast<unsigned int>(pow(8.f, static_cast<float>(i))) % 8];

        // make the octant position 1D for the linear memory
        nodeOffset = nextOctant.x + 2*nextOctant.y + 4*nextOctant.z;

        unsigned int pointer = nodePool[nodeOffset+childPointer*8].nodeTilePointer;
        __syncthreads();    //make sure all threads have a valid nodeTilePointer
        unsigned int reserve = getBit(pointer,31);

        if(reserve == 1)
        {
            // increment the global nodecount and allocate the memory in our
            unsigned int adress = atomicAdd(counter,1)+1;

            pointer = (adress & 0x3fffffff) | pointer;

            // set the divide flag to 1. this indicates that the child pointer is valid
            setBit(pointer,32);

            // make sure we don't reserve the same nodeTile next time :)
            unSetBit(pointer,31);

            nodePool[nodeOffset+childPointer*8].nodeTilePointer = pointer;
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

cudaError_t updateBrickPool(cudaArray_t &brickPool, dim3 textureDim)
{
    cudaError_t errorCode;

    cudaChannelFormatDesc channelDesc;
    errorCode = cudaGetChannelDesc(&channelDesc, brickPool);

    if(errorCode != cudaSuccess)
        return errorCode;

    errorCode = cudaBindSurfaceToArray(&surfRef, brickPool, &channelDesc);

    if(errorCode != cudaSuccess)
        return errorCode;

    dim3 block_dim(4, 4, 4);
    dim3 grid_dim(textureDim.x/block_dim.x, textureDim.y/block_dim.y, textureDim.z/block_dim.z);
    testFilling<<<grid_dim, block_dim>>>(textureDim);

    return cudaSuccess;
}

cudaError_t updateNodePool(uchar4* colorBufferDevPointer, node *nodePool, int poolSize)
{
    cudaError_t errorCode = cudaSuccess;
    int threadsPerBlock = 64;
    int blockCount = poolSize / threadsPerBlock;


    struct node *node_h = (struct node*)malloc(sizeof(node) * poolSize);

    errorCode = cudaMemcpy(node_h, nodePool, sizeof(node) * poolSize, cudaMemcpyDeviceToHost);

    if(errorCode != cudaSuccess)
        return errorCode;


    free(node_h);

    return cudaSuccess;
}

cudaError_t copyNodePoolToConstantMemory(node *nodePool, int poolSize)
{
    cudaError_t errorCode = cudaMemcpyToSymbol(constNodePool,nodePool,sizeof(node)*poolSize,0,cudaMemcpyDeviceToDevice);

    if(errorCode != cudaSuccess)
    {
        constantMemoryValid = false;
        return errorCode;
    }
    else
    {
        constantMemoryValid = true;
        return errorCode;
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
    int maxLevel = static_cast<int>(log((volumeResolution*volumeResolution*volumeResolution)/27)/log(8));

    printf("max level: %d \n", maxLevel);

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

    printf("counter: %d\n", *h_counter);

    for(int i=0;i<maxLevel;i++)
    {
        markNodeForSubdivision<<<blockCount, threadsPerBlock>>>(nodePool, poolSize, i, positionDevPointer, 1);
        cudaDeviceSynchronize();
        unsigned int maxNodes = static_cast<unsigned int>(pow(8,i));

        const int threadPerBlockReserve = 32;
        int blockCountReserve = maxNodes;

        if(maxNodes >= threadPerBlockReserve)
            blockCountReserve = maxNodes / threadPerBlockReserve;

        reserveMemoryForNodes <<< blockCountReserve, threadPerBlockReserve >>> (nodePool, maxNodes, i, d_counter);
        printf("memory reserved level %d\n", i);
        cudaDeviceSynchronize();

        cudaMemcpy(h_counter, d_counter, sizeof(unsigned int), cudaMemcpyDeviceToHost);

        printf("reserved node tiles: %d\n", *h_counter);
    }

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
    printf("memory cleared\n");

    return errorCode;
}