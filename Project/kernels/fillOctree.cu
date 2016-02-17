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

    return errorCode;
}

__global__
void clearNodePoolKernel(unsigned int *nodePool, int poolSize)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if(i >= poolSize)
        return;

    nodePool[i] = 0;
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

// just for debugging
__device__ void makeBrickWhite(const uint3 &brickCoords)
{
    surf3Dwrite(make_uchar4(255,255,255,255), colorBrickPool, brickCoords.x*sizeof(uchar4), brickCoords.y, brickCoords.z);
}

__global__ void filterBrickCornersFast(node* nodePool, unsigned int level)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    // make sure our index matches the node-adresses in a given octree level
    index += (constLevelIntervalMap[level].start*8);

    if(index >= constLevelIntervalMap[level].end*8)
        return;

    // load the target node that should be filled by mipmapping
    node targetNode = nodePool[index];
    __syncthreads();
  //  if(getBit(targetNode.value,31) != 0)
   // {
        //printf("index:%d x:%d y:%d z:%d, brickValue:%d startadress:%d end:%d\n", index,
          //     decodeBrickCoords(targetNode.value).x, decodeBrickCoords(targetNode.value).y,
            //   decodeBrickCoords(targetNode.value).z, targetNode.nodeTilePointer & 0x3fffffff, startAdress, endAdress);

        filterBrick(decodeBrickCoords(nodePool[index].value));
        //makeBrickWhite(decodeBrickCoords(nodePool[index].value)); // upper left corner. just for debugging
    //}

    __syncthreads();
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
        __syncthreads();
        childPointer = nodeTile & 0x3fffffff;

        if(i!=0)
        {
            position.x = 2 * position.x - nextOctant.x;
            position.y = 2 * position.y - nextOctant.y;
            position.z = 2 * position.z - nextOctant.z;
        }
    }

    // now we fill the corners of our bricks at the last level. This level is represented with 8 values inside a brick
    value = nodePool[offset].value;

  //  if(index < 10000)
    //    printf("ADRESSE: %d %d\n", offset, maxLevel);
    // we have a valid brick => fill it
    fillBrickCorners(decodeBrickCoords(value), position, colorBufferDevPointer[index]);
    setBit(value, 31);

    __syncthreads();
    nodePool[offset].value = value;
}

__global__ void fillNeighbours(node* nodePool, neighbours* neighbourPool, uint1* positionBuffer, unsigned int poolSize, unsigned int fragmentListSize, unsigned int level)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if(index >= fragmentListSize)
        return;

        float3 position;
        getVoxelPositionUINTtoFLOAT3(positionBuffer[index].x,position);

        float stepSize = 1.f/powf(2,level);// for some reason this is faster than lookups :D

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
        unsigned int nodeAdress = traverseToCorrespondingNode(nodePool, position, nodeLevel, level);


    //if(index < 10000 && level == 2)
      //  printf("ADRESSE: %d %d\n", nodeAdress, nodeLevel);

        // traverse to neighbours
        if (position.x + stepSize < 1)
        {
            // handle X
            float3 tmp = position;
            tmp.x += stepSize;

            X = traverseToCorrespondingNode(nodePool, tmp, foundOnLevel, level);

            if(!(X > (constLevelIntervalMap[level].start)*8 && X < (constLevelIntervalMap[level].end)*8))
            {
                X = 0;
               // WE HAVE NO NEIGHBOUR IN +X
            }
        }
        if (position.y + stepSize < 1)
        {
            // handle Y
            float3 tmp = position;
            tmp.y += stepSize;
            Y = traverseToCorrespondingNode(nodePool, tmp, foundOnLevel, level);

            if(!(Y > (constLevelIntervalMap[level].start)*8 && Y < (constLevelIntervalMap[level].end)*8))
            {
                Y = 0;
            }
        }
        if (position.z + stepSize < 1)
        {
            // handle Z
            float3 tmp = position;
            tmp.z += stepSize;
            Z = traverseToCorrespondingNode(nodePool, tmp, foundOnLevel, level);

            if(!(Z > (constLevelIntervalMap[level].start)*8 && Z < (constLevelIntervalMap[level].end)*8))
            {
                Z = 0;
            }
        }

        if (position.x - stepSize > 0)
        {
            // handle negX
            float3 tmp = position;
            tmp.x -= stepSize;
            negX = traverseToCorrespondingNode(nodePool, tmp, foundOnLevel, level);

            if(!(negX > (constLevelIntervalMap[level].start)*8 && negX < (constLevelIntervalMap[level].end)*8))
            {
                negX = 0;
            }
        }
        if (position.y - stepSize > 0)
        {
            // handle negY
            float3 tmp = position;
            tmp.y -= stepSize;
            negY = traverseToCorrespondingNode(nodePool, tmp, foundOnLevel, level);

            if(!(negY > (constLevelIntervalMap[level].start)*8 && negY < (constLevelIntervalMap[level].end)*8))
            {
                negY = 0;
            }
        }
        if (position.z - stepSize > 0)
        {
            // handle negZ
            float3 tmp = position;
            tmp.z -= stepSize;
            negZ = traverseToCorrespondingNode(nodePool, tmp, foundOnLevel, level);

            if(!(negZ > (constLevelIntervalMap[level].start)*8 && negZ < (constLevelIntervalMap[level].end)*8))
            {
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

__global__ void reserveMemoryForNodesFast(node* nodePool, unsigned int startAdress, unsigned int maxNodesOnLevel, unsigned int *counter, unsigned int brickPoolResolution, unsigned int brickResolution, unsigned int lastLevel, unsigned int poolSize)
{
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;

    if(index >= maxNodesOnLevel || index > poolSize)
        return;

    unsigned int nodeAdress = 0;

    if(maxNodesOnLevel != 1)
        nodeAdress = (startAdress+1)*8 + index;

    unsigned int pointer = nodePool[nodeAdress].nodeTilePointer;
    unsigned int value = nodePool[nodeAdress].value;

    __syncthreads();

    if (getBit(pointer, 31) == 1)
    {
        // increment the global nodecount and allocate the memory in our
        unsigned int adress = atomicAdd(counter, 1) + 1;

        pointer = (adress & 0x3fffffff) | pointer;
        value = encodeBrickCoords(getBrickCoords(adress-1, brickPoolResolution, brickResolution));

        // set the first bit to 1. this indicates, that we use the texture brick instead of a constant value as color.
        setBit(value, 32);
        setBit(pointer, 32);

        // make sure we don't reserve the same nodeTile next time :)
        unSetBit(pointer, 31);

        if(lastLevel == 1)
            unSetBit(pointer,32);

        nodePool[nodeAdress].nodeTilePointer = pointer;
        nodePool[nodeAdress].value = value;
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

            pointer = (adress & 0x3fffffff) | pointer;
            value = encodeBrickCoords(getBrickCoords(adress, brickPoolResolution, brickResolution));
            uint3 test = decodeBrickCoords(value);

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
    int maxLevel = static_cast<int>(log((volumeResolution*volumeResolution*volumeResolution))/log(8));
    // note that we dont calculate +1 as we store 8 voxels per brick

    dim3 grid_dim(volumeResolution/block_dim.x,volumeResolution/block_dim.y,volumeResolution/block_dim.z);

    int blockCount = fragmentListSize / threadsPerBlockFragmentList + 1;

    errorCode = cudaBindSurfaceToArray(colorBrickPool, brickPool);

    unsigned int *h_counter = new unsigned int[1];
    *h_counter = 0;

    cudaMemcpy(d_counter,h_counter,sizeof(unsigned int),cudaMemcpyHostToDevice);

    clearBrickPool<<<grid_dim, block_dim>>>(volumeResolution);

    int lastLevel = 0;
    unsigned int reservedOld = 0;

    for(int i=0;i<maxLevel;i++)
    {
        // todo: this is silly :D
        if(i!=0)
            LevelIntervalMap[i-1].start = reservedOld+1;

        markNodeForSubdivision<<<blockCount, threadsPerBlockFragmentList>>>(nodePool, poolSize, i, positionDevPointer, fragmentListSize);
        unsigned int maxNodes = static_cast<unsigned int>(pow(8,i));

        int blockCountReserve = maxNodes;

        if(maxNodes >= threadPerBlockReserve)
            blockCountReserve = maxNodes / threadPerBlockReserve+1;

        if(i == maxLevel-1)
            lastLevel = 1;

        cudaMemcpy(h_counter, d_counter, sizeof(unsigned int), cudaMemcpyDeviceToHost);
        reserveMemoryForNodesFast <<< blockCountReserve, threadPerBlockReserve >>> (nodePool, reservedOld, maxNodes, d_counter, volumeResolution, 3, lastLevel, poolSize);

            // remember counter
        reservedOld = *h_counter;

        // todo: this is silly
        if(i!=0)
            LevelIntervalMap[i-1].end = reservedOld-1;
    }

    // make sure we fill the interval map complete
    cudaMemcpy(h_counter, d_counter, sizeof(unsigned int), cudaMemcpyDeviceToHost);
    LevelIntervalMap[maxLevel-1].start = LevelIntervalMap[maxLevel-2].end+1;
    LevelIntervalMap[maxLevel-1].end = *h_counter-1;

    // copy the level interval map to constant memory
    errorCode = cudaMemcpyToSymbol(constLevelIntervalMap, LevelIntervalMap, sizeof(LevelInterval)*10);

    cudaDeviceSynchronize();
    insertVoxelsInLastLevel<<<blockCount,threadsPerBlockFragmentList>>>(nodePool,positionDevPointer,colorBufferDevPointer,maxLevel, fragmentListSize);

    unsigned int nodeCount = static_cast<unsigned int>(pow(8,maxLevel-1));

    cudaDeviceSynchronize();

    for(int i=1;i<7;i++)
        fillNeighbours << < blockCount, threadsPerBlockFragmentList >> > (nodePool, neighbourPool, positionDevPointer, poolSize, fragmentListSize, i);

    const int level = 6;
    unsigned int tmpBlock = ((LevelIntervalMap[level].end-LevelIntervalMap[level].start)*8) / threadPerBlockSpread + 1;

   // printf("LEVEL %d start: %d end:%d\n", 5, LevelIntervalMap[5].start*8, LevelIntervalMap[5].end*8);

    // filter the last level with an inverse gaussian kernel
	
	filterBrickCornersFast<<<tmpBlock,threadPerBlockSpread>>>(nodePool,level);

    unsigned int combineBlockCount = static_cast<unsigned int>(pow(8,maxLevel-1)) / threadsPerBlockCombineBorders;
    combineBrickBordersFast<<<tmpBlock, threadPerBlockSpread>>>(nodePool, neighbourPool, level);
    cudaDeviceSynchronize();

    // MIPMAP we have some crap with the 0 level. therefore we subtract 3 :)
    for(int i=maxLevel-3;i>=0;i--)
    {
        unsigned int blockCountMipMap = 1;
        unsigned int intervalWidth = (LevelIntervalMap[i].end - LevelIntervalMap[i].start)*8;

        if(threadsPerBlockMipMap < intervalWidth)
            blockCountMipMap = intervalWidth / threadsPerBlockMipMap+1;

        mipMapOctreeLevel<<<blockCountMipMap,threadsPerBlockMipMap>>>(nodePool, i);
        cudaDeviceSynchronize();
        printf("i %d\n",i);
        combineBrickBordersFast<<<tmpBlock, threadPerBlockSpread>>>(nodePool, neighbourPool, i);
        cudaDeviceSynchronize();
    }

    delete h_counter;

    return errorCode;
}

cudaError_t clearNodePoolCuda(node *nodePool, neighbours* neighbourPool, int poolSize)
{
    cudaError_t errorCode = cudaSuccess;

    int blockCount = (poolSize*2) / threadsPerBlockClear;
    int neighbourPoolBlockCount = (poolSize*6) / threadsPerBlockClear;

    // clear the nodepool
    clearNodePoolKernel<<<neighbourPoolBlockCount, threadsPerBlockClear>>>(reinterpret_cast<unsigned int*>(neighbourPool), poolSize*6);
    clearNodePoolKernel<<<blockCount, threadsPerBlockClear>>>(reinterpret_cast<unsigned int*>(nodePool), poolSize*2);

    return errorCode;
}

cudaError_t initMemory()
{
    cudaError_t error = cudaSuccess;
    //
    error = cudaMalloc(&d_counter, sizeof(int));

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

    LevelInterval *intervalMap = new LevelInterval[10];

    for(int i=0;i<10;i++)
    {
        intervalMap[i].start = 0;
        intervalMap[i].end = 0;
    }

    uint3 *indexLookUp = new uint3[27];

    for(unsigned int i=0;i<27;i++)
    {
        indexLookUp[i].x = i / 9;
        indexLookUp[i].y = (i / 3) % 3;
        indexLookUp[i].z = i % 3;
    }

    error = cudaMemcpyToSymbol(lookup_octants, octants, sizeof(uint3)*8);
    error = cudaMemcpyToSymbol(insertPositions, insertpos, sizeof(uint3)*8);
    error = cudaMemcpyToSymbol(constLevelIntervalMap, intervalMap, sizeof(LevelInterval)*10);
    error = cudaMemcpyToSymbol(constLookUp1Dto3DIndex, indexLookUp, sizeof(uint3)*27);

    delete[] octants;
    delete[] insertpos;
    delete[] intervalMap;
    delete[] indexLookUp;

    return error;
}

cudaError_t freeMemory()
{
    return cudaFree(d_counter);
}