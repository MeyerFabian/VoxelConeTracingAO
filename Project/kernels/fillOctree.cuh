#include <driver_types.h>
#include <vector_types.h>
#include <SparseOctree/NodePool.h>

extern "C" // this is not necessary imho, but gives a better idea on where the function comes from
{
// clears the node pool (all bits are set to 0)
    cudaError_t clearNodePoolCuda(node *nodePool, int poolSize);

// builds the whole SVO (nodepool and brickpool) by using a fragmentlist (includes filtering and mipmapping)
    cudaError_t buildSVO(node *nodePool,
                         int poolSize,
                         cudaArray_t *brickPool,
                         dim3 textureDim,
                         uint1* positionDevPointer,
                         uchar4* colorBufferDevPointer,
                         uchar4* normalDevPointer,
                         int fragmentListSize);

// sets the volume resolution within the constant memory
    cudaError_t setVolumeResulution(int resolution);
}
