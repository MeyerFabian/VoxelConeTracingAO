#include <driver_types.h>
#include <vector_types.h>
#include <SparseOctree/NodePool.h>

extern "C" // this is not necessary imho, but gives a better idea on where the function comes from
{
    cudaError_t updateBrickPool(cudaArray_t &brickPool, dim3 textureDim);   // hier muss noch der nodepool und die voxelliste hin
    cudaError_t updateNodePool(cudaArray_t &voxel, node *nodePool, int poolSize);        // hier muss noch der nodepool hin
}
