#include <stdio.h>
#include <cuda_runtime.h>
#include "fillOctree.cuh"


const int maxNodePoolSize = 1024;

bool constantMemoryValid = false;   // the flag indicates wheather a kernel is allowed to use the constantNodePool
__constant__ node constNodePool[maxNodePoolSize];
__constant__ int constVolumeResolution[1];
__device__ int globalNodePoolCounter = 0;

surface<void, cudaSurfaceType3D> surfRef;

cudaError_t setVolumeResulution(int resolution)
{
    cudaError_t errorCode = cudaMemcpyToSymbol(constVolumeResolution, &resolution, sizeof(int));
    return errorCode;
}

__device__
int getBits(unsigned int value, int start, int quantity)
{
    const unsigned int mask_bits = 0xffffffff;

    assert(start <= 31);
    if (start > 31)
        return 0;

    if(quantity > 32-start)
        quantity = 32-start;

    return (value >> start) & (mask_bits >> (32 - quantity));
}

__device__
int getBit(unsigned int value, int position)
{
    return (value >> position-1) & 1;
}

__device__
void setBit(unsigned int &value, int position)
{
    value |= (1u << (position-1));
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
void testNodeFilling(node *nodePool, int poolSize, uchar4* colorBufferDevPointer)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if(i >= poolSize)
    {
        return;
    }

    if(i == 0)
        printf("%d ,%d, %d, %d \n",colorBufferDevPointer[0].x, colorBufferDevPointer[0].y, colorBufferDevPointer[0].z, colorBufferDevPointer[0].w);

    nodePool[i].nodeTilePointer = 10;
    nodePool[i].value = getBits(nodePool[i].nodeTilePointer,31,1);
}

__global__ void markNodeForSubdivision(node *nodePool, int poolSize, int maxLevel, uint1* positionBuffer, int volumeSideLength)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    uint codedPosition = positionBuffer[index].x;
    float3 position;
    position.x = getBits(codedPosition,2,10)  / constVolumeResolution[0];
    position.y = getBits(codedPosition,12,10) / constVolumeResolution[0];
    position.z = getBits(codedPosition,22,10) / constVolumeResolution[0];

    unsigned int nodeOffset = 0;
    unsigned int childPointer = 0;

    for(int i=0;i<maxLevel;i++)
    {
        uint3 nextOctant = make_uint3(0, 0, 0);
        // determine octant for the given voxel
        nextOctant.x = static_cast<unsigned int>(2 * position.x);
        nextOctant.y = static_cast<unsigned int>(2 * position.y);
        nextOctant.z = static_cast<unsigned int>(2 * position.z);

        // make the octant position 1D for the linear memory
        nodeOffset = nextOctant.x + 2*nextOctant.y + 4*nextOctant.z;

        unsigned int maxDivide = getBit(nodePool[nodeOffset+childPointer].nodeTilePointer,32);
        if(maxDivide == 0)
        {
            // set second bit to 1
            setBit(nodePool[nodeOffset+childPointer].nodeTilePointer,31);
            break;
        }
        else
        {
            // traverse further
            childPointer = getBits(nodePool[nodeOffset + childPointer].nodeTilePointer, 2, 30);
        }

        position.x = 2*position.x - nextOctant.x;
        position.y = 2*position.y - nextOctant.y;
        position.z = 2*position.z - nextOctant.z;
    }


}

__global__ void reserveMemoryForNodes(node *nodePool, int poolSize, int level)
{
    int indexX = blockIdx.x * blockDim.x + threadIdx.x;
    int indexY = blockIdx.y * blockDim.y + threadIdx.y;
    int indexZ = blockIdx.z * blockDim.z + threadIdx.z;

    float3 position;
    // make sure we traverse all nodes
    position.x = indexX  / constVolumeResolution[0];
    position.y = indexY / constVolumeResolution[0];
    position.z = indexZ / constVolumeResolution[0];

    unsigned int nodeOffset = 0;
    unsigned int childPointer = 0;

    for(int i=0;i<level;i++)
    {
        uint3 nextOctant = make_uint3(0, 0, 0);
        // determine octant for the given voxel
        nextOctant.x = static_cast<unsigned int>(2 * position.x);
        nextOctant.y = static_cast<unsigned int>(2 * position.y);
        nextOctant.z = static_cast<unsigned int>(2 * position.z);

        // make the octant position 1D for the linear memory
        nodeOffset = nextOctant.x + 2*nextOctant.y + 4*nextOctant.z;

        unsigned int reserve = getBit(nodePool[nodeOffset+childPointer].nodeTilePointer,31);
        if(reserve == 1)
        {
            // increment the global nodecount and allocate the memory in our
            int adress = atomicAdd(&globalNodePoolCounter,1);

            // TODO: reserve memory
            // TODO: increase counter
            // TODO: set child pointer of node
            setBit(nodePool[nodeOffset+childPointer].nodeTilePointer,32);
            break;
        }
        else
        {
            // traverse further
            childPointer = getBits(nodePool[nodeOffset + childPointer].nodeTilePointer, 2, 30);
        }

        position.x = 2*position.x - nextOctant.x;
        position.y = 2*position.y - nextOctant.y;
        position.z = 2*position.z - nextOctant.z;
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
    int maxLevel = 2;
    dim3 block_dim(32, 0, 0);
    dim3 grid_dim(fragmentListSize/block_dim.x, 0, 0);

    int threadsPerBlock = 64;
    int blockCount = fragmentListSize / threadsPerBlock;

    for(int i=0;i<6;i++)
    {
        markNodeForSubdivision<<<blockCount, threadsPerBlock>>>(nodePool, poolSize, maxLevel, positionDevPointer, 1);
        cudaDeviceSynchronize();
        unsigned int maxNodes = static_cast<unsigned int>(pow(8,i));
        dim3 nodes(maxNodes,maxNodes,maxNodes);
        // reserve memory
        dim3 block_dim_memory(4, 4, 4);
        dim3 grid_dim_memory(1,1,1);
        if(maxNodes >= 8)
            grid_dim_memory = dim3(nodes.x/block_dim_memory.x, nodes.y/block_dim_memory.y, nodes.z/block_dim_memory.z);

        // start for every possible node in this level a thread. this way we make sure, that we dont miss one
        //reserveMemoryForNodes<<<grid_dim_memory, block_dim_memory>>>(nodePool, poolSize, i);
        //cudaDeviceSynchronize();
    }

}