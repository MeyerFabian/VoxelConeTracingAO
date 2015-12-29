#include <stdio.h>
#include <cuda_runtime.h>
#include "fillOctree.cuh"


const int maxNodePoolSize = 1024;

bool constantMemoryValid = false;   // the flag indicates wheather a kernel is allowed to use the constantNodePool
__constant__ node constNodePool[maxNodePoolSize];

surface<void, cudaSurfaceType3D> surfRef;

__device__
int getBits(int value, int start, int quantity)
{
    const unsigned int mask_bits = 0xffffffff;

    assert(start <= 31);
    if (start > 31)
        return 0;

    if(quantity > 32-start)
        quantity = 32-start;

    return (value >> start) & (mask_bits >> (32 - quantity));
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
void testNodeFilling(nodeTile *nodePool, int poolSize, uchar4* colorBufferDevPointer)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if(i >= poolSize)
    {
        return;
    }

    if(i == 0)
        printf("%d ,%d, %d, %d \n",colorBufferDevPointer[0].x, colorBufferDevPointer[0].y, colorBufferDevPointer[0].z, colorBufferDevPointer[0].w);

    nodePool[i].node1.nodeTilePointer = 10;
    nodePool[i].node1.value = getBits(nodePool[i].node1.nodeTilePointer,31,1);
}

__global__ void markNodeForSubdivision(nodeTile *nodePool, int poolSize, int maxLevel, uint1* positionBuffer, int volumeSideLength)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;


    uint codedPosition = positionBuffer[index].x;
    float3 position;
    position.x = getBits(codedPosition,2,10);
    position.y = getBits(codedPosition,12,10);
    position.z = getBits(codedPosition,22,10);
    int nextIndex = 0;
    // TODO: rethink for tommorow
    for(int i=0;i<maxLevel;i++)
    {
        // calculate index
        if(getBits(nodePool[nextIndex].node1.nodeTilePointer,1,1) == 1)
        {
            nextIndex = getBits(nodePool[nextIndex].node1.nodeTilePointer,2,31);
            // visited. traverse further down
        }
        else
        {
            // not visited. mark as visited

            // exit for loop and kernel
            break;
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

cudaError_t updateNodePool(uchar4* colorBufferDevPointer, nodeTile *nodePool, int poolSize)
{
    cudaError_t errorCode = cudaSuccess;
    int threadsPerBlock = 64;
    int blockCount = poolSize / threadsPerBlock;

    testNodeFilling<<<blockCount, threadsPerBlock>>>(nodePool, poolSize, colorBufferDevPointer);

    struct node *node_h = (struct node*)malloc(sizeof(node) * poolSize);

    errorCode = cudaMemcpy(node_h, nodePool, sizeof(node) * poolSize, cudaMemcpyDeviceToHost);

    if(errorCode != cudaSuccess)
        return errorCode;

/*
    for(int i=0;i<poolSize;i++)
        printf("%d, %d \n",node_h[i].nodeTilePointer,node_h[i].value);
*/

    free(node_h);

    return cudaSuccess;
}

cudaError_t copyNodePoolToConstantMemory(nodeTile *nodePool, int poolSize)
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

cudaError_t buildSVO(nodeTile *nodePool,
                     int poolSize,
                     cudaArray_t &brickPool,
                     dim3 textureDim,
                     uint1* positionDevPointer,
                     uchar4* colorBufferDevPointer,
                     uchar4* normalDevPointer,
                     int fragmentListSize)
{

}