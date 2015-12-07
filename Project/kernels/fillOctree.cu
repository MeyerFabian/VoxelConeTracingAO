#include <stdio.h>
#include <cuda_runtime.h>
#include "fillOctree.cuh"


const int maxNodePoolSize = 8192;

bool constantMemoryValid = false;   // the flag indicates wheather a kernel is allowed to use the constantNodePool
__constant__ int constNodePool[16384];

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
void testNodeFilling(node *nodePool, int poolSize)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if(i >= poolSize)
    {
        return;
    }

    nodePool->nodeTilePointer[i] = 10;
    nodePool->value[i] = 10;
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

cudaError_t updateNodePool(cudaArray_t &voxel, node *nodePool, int poolSize)
{
    cudaError_t errorCode = cudaSuccess;
    int threadsPerBlock = 16;
    int blockCount = poolSize / threadsPerBlock;

    testNodeFilling<<<blockCount, threadsPerBlock>>>(nodePool, poolSize);

    struct node *node_h = (struct node*)malloc(sizeof(node) + sizeof(int)*poolSize*2);

    errorCode = cudaMemcpy(node_h, nodePool, sizeof(node) + sizeof(int)*poolSize*2, cudaMemcpyDeviceToHost);

    if(errorCode != cudaSuccess)
        return errorCode;

/*
    for(int i=0;i<poolSize;i++)
        printf("%d, %d \n",node_h->nodeTilePointer[i],node_h->value[i]);
*/

    free(node_h);

    return cudaSuccess;
}

cudaError_t copyNodePoolToConstantMemory(node *nodePool, int poolSize)
{
    cudaError_t errorCode = cudaMemcpyToSymbol(constNodePool,nodePool,sizeof(node) + sizeof(int)*poolSize*2,0,cudaMemcpyDeviceToDevice);

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