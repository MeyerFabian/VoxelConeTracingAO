#include <driver_types.h>
#include <vector_types.h>
#include <SparseOctree/NodePool.h>

extern "C" // this is not necessary imho, but gives a better idea on where the function comes from
{
    cudaError_t updateBrickPool(cudaArray_t &brickPool, dim3 textureDim);   // hier muss noch der nodepool und die voxelliste hin
    cudaError_t updateNodePool(uchar4* colorBufferDevPointer, node *nodePool, int poolSize);        // hier muss noch der nodepool hin
    cudaError_t copyNodePoolToConstantMemory(node *nodePool, int poolSize);
    cudaError_t clearNodePoolCuda(node *nodePool, int poolSize);

    cudaError_t buildSVO(node *nodePool,
                         int poolSize,
                         cudaArray_t *brickPool,
                         dim3 textureDim,
                         uint1* positionDevPointer,
                         uchar4* colorBufferDevPointer,
                         uchar4* normalDevPointer,
                         int fragmentListSize);

cudaError_t setVolumeResulution(int resolution);
}
