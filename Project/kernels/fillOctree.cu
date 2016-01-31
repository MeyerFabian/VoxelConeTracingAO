#include <stdio.h>
#include <cuda_runtime.h>

#include "globalResources.cuh"
#include "fillOctree.cuh"
#include "bitUtilities.cuh"
#include "octreeMipMapping.cuh"
#include "brickUtilities.cuh"
#include "traverseKernels.cuh"


cudaError_t setVolumeResulution(int resolution)
{
    cudaError_t errorCode = cudaSuccess;
    volumeResolution = resolution;

    uint3 *octants = new uint3[8];
    octants[0] = make_uint3(0,0,0);
    octants[1] = make_uint3(0,0,1);
    octants[2] = make_uint3(0,1,0);
    octants[3] = make_uint3(0,1,1);
    octants[4] = make_uint3(1,0,0);
    octants[5] = make_uint3(1,0,1);
    octants[6] = make_uint3(1,1,0);
    octants[7] = make_uint3(1,1,1);

    uint3 *insertpos = new uint3[8];
    // front corners
    insertpos[0] = make_uint3(0,0,0);
    insertpos[1] = make_uint3(2,0,0);
    insertpos[2] = make_uint3(0,2,0);
    insertpos[3] = make_uint3(2,2,0);

    //back corners
    insertpos[4] = make_uint3(0,0,2);
    insertpos[5] = make_uint3(2,0,2);
    insertpos[6] = make_uint3(0,2,2);
    insertpos[7] = make_uint3(2,2,2);

    errorCode = cudaMemcpyToSymbol(lookup_octants, octants, sizeof(uint3)*8);
    errorCode = cudaMemcpyToSymbol(insertPositions, insertpos, sizeof(uint3)*8);

    delete[] octants;
    delete[] insertpos;

    return errorCode;
}

__global__
void clearNodePoolKernel(node *nodePool, neighbours* neighbourPool, int poolSize)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if(i >= poolSize)
        return;

    nodePool[i].nodeTilePointer = 0;
    nodePool[i].value = 0;

    neighbourPool[i].negX = 0;
    neighbourPool[i].negY = 0;
    neighbourPool[i].negZ = 0;

    neighbourPool[i].X = 0;
    neighbourPool[i].Y = 0;
    neighbourPool[i].Z = 0;
}

__global__
void clearBrickPool(unsigned int brick_res)
{
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;
    int z = blockIdx.z*blockDim.z + threadIdx.z;

   if(x >= brick_res || y >= brick_res || z >= brick_res)
        return;

    surf3Dwrite(make_uchar4(0, 0, 0, 0), colorBrickPool, x*sizeof(uchar4), y, z);
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

    uint3 nextOctant;
    unsigned int octantIdx = 0;

    for (int i = 0; i < maxLevel; i++)
    {
        if(i==0)
            octantIdx = 0;
        else // here we make sure that we are able to reach every node within the tree
            octantIdx = (index / static_cast<unsigned int>(pow(8.f, static_cast<float>(i-1)))) % 8;

        nextOctant = lookup_octants[octantIdx];

        // make the octant position 1D for the linear memory
        nodeOffset = nextOctant.x + 2 * nextOctant.y + 4 * nextOctant.z;
        offset = nodeOffset + childPointer * 8;

        unsigned int pointer = nodePool[offset].nodeTilePointer;

        // traverse further until we reach a non valid point. as we wish to iterate to the bottom, we return if there is an invalid connectioon (should not happen)
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

    float3 position;
    getVoxelPositionUINTtoFLOAT3(positionBuffer[index].x,position);

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
    unsigned int foundOn=0;
    //unsigned int offset = traverseToCorrespondingNode(nodePool,position,foundOn,maxLevel);
    // now we fill the corners of our bricks at the last level. This level is represented with 8 values inside a brick
    value = nodePool[offset].value;

    if(getBit(value,32) == 1)
    {
        // we have a valid brick => fill it
        fillBrickCorners(decodeBrickCoords(value), position, colorBufferDevPointer[index]);
        setBit(value,31);
        __syncthreads();
        nodePool[offset].value = value;
    }
}

__global__ void fillNeighbours(node* nodePool, neighbours* neighbourPool, uint1* positionBuffer, unsigned int poolSize, unsigned int fragmentListSize, unsigned int level)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if(index >= fragmentListSize)
        return;

    float3 position;
    getVoxelPositionUINTtoFLOAT3(positionBuffer[index].x,position);

    for(int i=1;i<level;i++)
    {
        float stepSize = 1.f/powf(2,i);// for some reason this is faster than lookups :D

        // initialise all neighbours to no neighbour :P
        unsigned int X = 0;
        unsigned int Y = 0;
        unsigned int Z = 0;
        unsigned int negX = 0;
        unsigned int negY = 0;
        unsigned int negZ = 0;

        unsigned int nodeLevel = 0;
        unsigned int foundOnLevel = 0;

        // traverse to my node // TODO: it might be easier if we stack the parent level
        unsigned int nodeAdress = traverseToCorrespondingNode(nodePool, position, nodeLevel, i);

        // traverse to neighbours
        if (position.x + stepSize < 1) {
            // handle X
            float3 tmp = position;
            tmp.x += stepSize;
            X = traverseToCorrespondingNode(nodePool, tmp, foundOnLevel, i);

            if (nodeLevel != foundOnLevel) {
                X = 0;
            }
        }
        if (position.y + stepSize < 1) {
            // handle Y
            float3 tmp = position;
            tmp.y += stepSize;
            Y = traverseToCorrespondingNode(nodePool, tmp, foundOnLevel, i);

            if (nodeLevel != foundOnLevel) {
                Y = 0;
            }
        }
        if (position.z + stepSize < 1) {
            // handle Z
            float3 tmp = position;
            tmp.z += stepSize;
            Z = traverseToCorrespondingNode(nodePool, tmp, foundOnLevel, i);

            if (nodeLevel != foundOnLevel) {
                Z = 0;
            }
        }

        if (position.x - stepSize > 0) {
            // handle negX
            float3 tmp = position;
            tmp.x -= stepSize;
            negX = traverseToCorrespondingNode(nodePool, tmp, foundOnLevel, i);

            if (nodeLevel != foundOnLevel) {
                negX = 0;
            }
        }
        if (position.y - stepSize > 0) {
            // handle negY
            float3 tmp = position;
            tmp.y -= stepSize;
            negY = traverseToCorrespondingNode(nodePool, tmp, foundOnLevel, i);

            if (nodeLevel != foundOnLevel) {
                negY = 0;
            }
        }
        if (position.z - stepSize > 0) {
            // handle negZ
            float3 tmp = position;
            tmp.z -= stepSize;
            negZ = traverseToCorrespondingNode(nodePool, tmp, foundOnLevel, i);

            if (nodeLevel != foundOnLevel) {
                negZ = 0;
            }
        }

        __syncthreads();// probably not necessary

        neighbourPool[nodeAdress].X = X;
        neighbourPool[nodeAdress].Y = Y;
        neighbourPool[nodeAdress].Z = Z;
        neighbourPool[nodeAdress].negX = negX;
        neighbourPool[nodeAdress].negY = negY;
        neighbourPool[nodeAdress].negZ = negZ;
    }
}

// traverses the octree with one thread for each entry in the fragmentlist. Marks every node on its way as dividable
// this kernel gets executed successively for each level of the tree
__global__ void markNodeForSubdivision(node *nodePool, int poolSize, int maxLevel, uint1* positionBuffer, int fragmentListSize)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if(index >= fragmentListSize)
        return;

    float3 position;
    getVoxelPositionUINTtoFLOAT3(positionBuffer[index].x,position);

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
/*
    uint3 octants[8];
    octants[0] = make_uint3(0,0,0);
    octants[1] = make_uint3(0,0,1);
    octants[2] = make_uint3(0,1,0);
    octants[3] = make_uint3(0,1,1);
    octants[4] = make_uint3(1,0,0);
    octants[5] = make_uint3(1,0,1);
    octants[6] = make_uint3(1,1,0);
    octants[7] = make_uint3(1,1,1);
*/
    /*  The lookup version is slower. we have probably no registers left. const memory is slower as well
    unsigned int powLookup[8];
    powLookup[0] = 1;
    powLookup[1] = 8;
    powLookup[2] = 64;
    powLookup[3] = 512;
    powLookup[4] = 4096;
    powLookup[5] = 32768;
    powLookup[6] = 262144;
    powLookup[7] = 2097152;*/
    //static_cast<unsigned int>(powf(8.f, static_cast<float>(i-1)))

    uint3 nextOctant;
    unsigned int octantIdx = 0;

    for (int i = 0; i <=level; i++)
    {
        if(i==0)
            octantIdx = 0;
        else
            octantIdx = (index / static_cast<unsigned int>(powf(8.f, static_cast<float>(i-1)))) % 8;

        nextOctant = lookup_octants[octantIdx];

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
                     neighbours* neighbourPool,
                     int poolSize,
                     cudaArray *brickPool,
                     dim3 textureDim,
                     uint1* positionDevPointer,
                     uchar4* colorBufferDevPointer,
                     uchar4* normalDevPointer,
                     int fragmentListSize)
{
    cudaError_t errorCode = cudaSuccess;
    // calculate maxlevel
    int maxLevel = static_cast<int>(log((volumeResolution*volumeResolution*volumeResolution))/log(8)+1);
    // note that we dont calculate +1 as we store 8 voxels per brick

    dim3 block_dim(4,4,4);
    dim3 grid_dim(volumeResolution/block_dim.x,volumeResolution/block_dim.y,volumeResolution/block_dim.z);

    int threadsPerBlock = 64;
    int blockCount = fragmentListSize / threadsPerBlock;


    unsigned int *h_counter = new unsigned int[1];
    unsigned int *d_counter;
    *h_counter = 0;

    cudaMalloc(&d_counter, sizeof(int));
    cudaMemcpy(d_counter,h_counter,sizeof(unsigned int),cudaMemcpyHostToDevice);


    clearCounter<<<1,1>>>();
    clearBrickPool<<<grid_dim, block_dim>>>(volumeResolution);

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

        cudaMemcpy(h_counter, d_counter, sizeof(unsigned int), cudaMemcpyDeviceToHost);
        printf("reserved node tiles: %d\n", *h_counter);
    }

    // still not sure if this works
    errorCode = cudaMemcpyToSymbol(constNodePool, nodePool, sizeof(node)*maxNodePoolSizeForConstMemory,0,cudaMemcpyDeviceToDevice);

    //fillNeighbours <<< blockCount, threadsPerBlock >>> (nodePool, neighbourPool, positionDevPointer, poolSize, fragmentListSize, maxLevel);
    //cudaDeviceSynchronize();

    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc( 8, 8, 8, 8, cudaChannelFormatKindUnsigned );
    //errorCode = cudaGetChannelDesc(&channelDesc, *brickPool);
    errorCode = cudaBindSurfaceToArray(colorBrickPool, brickPool);

   // cudaDeviceSynchronize();
    insertVoxelsInLastLevel<<<blockCount,threadsPerBlock>>>(nodePool,positionDevPointer,colorBufferDevPointer,maxLevel, fragmentListSize);

    const unsigned int threadPerBlockSpread = 512;
    unsigned int blockCountSpread;
    unsigned int nodeCount = static_cast<unsigned int>(pow(8,maxLevel-1));

    blockCountSpread = nodeCount;

    if(nodeCount >= threadPerBlockSpread)
        blockCountSpread = nodeCount / threadPerBlockSpread;

    cudaDeviceSynchronize();
    //filterBrickCorners<<<blockCountSpread, threadPerBlockSpread>>>(nodePool, nodeCount, maxLevel);

    cudaFree(d_counter);
    delete h_counter;

    return errorCode;
}

cudaError_t clearNodePoolCuda(node *nodePool, neighbours* neighbourPool, int poolSize)
{
    cudaError_t errorCode = cudaSuccess;
    int threadsPerBlock = 64;
    int blockCount = poolSize / threadsPerBlock;

    // clear the nodepool
    clearNodePoolKernel<<<blockCount, threadsPerBlock>>>(nodePool, neighbourPool, poolSize);

    return errorCode;
}