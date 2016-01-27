//
// Created by nils1990 on 06.12.15.
//
#include "SparseVoxelOctree.h"
extern "C" // this is not necessary imho, but gives a better idea on where the function comes from
{
cudaError_t buildSVO(node *nodePool,
                     int poolSize,
                     cudaArray_t *brickPool,
                     dim3 textureDim,
                     uint1 *positionDevPointer,
                     uchar4 *colorBufferDevPointer,
                     uchar4 *normalDevPointer,
                     int fragmentListSize);
}

void SparseVoxelOctree::init()
{
    // testsing for node and brick-pool
    m_nodePool.init();
    m_brickPool.init();

    m_brickPool.registerTextureForCUDAWriting();
}
SparseVoxelOctree::~SparseVoxelOctree()
{
    m_brickPool.unregisterTextureForCUDA();
}

void SparseVoxelOctree::fillGui()
{

}

void SparseVoxelOctree::buildOctree(uint1 *positionFragmentList,uchar4 *colorFragmentList,uchar4 *normalFragmentList, int fragmentListSize)
{
    m_nodePool.mapToCUDA();
    m_brickPool.mapToCUDA();

    buildSVO(m_nodePool.getNodePoolDevicePointer(),
             m_nodePool.getPoolSize(),
             m_brickPool.getBrickPoolArray(),
             dim3(384,384,384),
             positionFragmentList,
             colorFragmentList,
             normalFragmentList,
             fragmentListSize);

    m_brickPool.unmapFromCUDA();
    m_nodePool.unmapFromCUDA();
}

void SparseVoxelOctree::clearOctree()
{
    m_nodePool.mapToCUDA();
    m_nodePool.clearNodePool();
    m_nodePool.unmapFromCUDA();
}
