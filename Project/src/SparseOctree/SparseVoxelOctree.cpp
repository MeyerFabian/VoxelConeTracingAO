//
// Created by nils1990 on 06.12.15.
//
#include "src/Utilities/errorUtils.h"
#include "SparseVoxelOctree.h"
extern "C" // this is not necessary imho, but gives a better idea on where the function comes from
{
    cudaError_t buildSVO(node *nodePool,
                         neighbours* neighbourPool,
                         int poolSize,
                         cudaArray *brickPool,
                         dim3 textureDim,
                         uint1 *positionDevPointer,
                         cudaArray *colorVolumeArray,
                         cudaArray *normalVolumeArray,
                         int fragmentListSize);

    cudaError_t setVolumeResulution(int resolution);
    cudaError_t initMemory();
    cudaError_t freeMemory();
}

void SparseVoxelOctree::init()
{
    // testsing for node and brick-pool
    m_nodePool.init();
    m_brickPool.init(256,256,256);
    cudaErrorCheck(setVolumeResulution(m_brickPool.getResolution().x));
    cudaErrorCheck(initMemory());

    m_brickPool.registerTextureForCUDAWriting();
}
SparseVoxelOctree::~SparseVoxelOctree()
{
    m_brickPool.unregisterTextureForCUDA();
    cudaErrorCheck(freeMemory());
}

void SparseVoxelOctree::fillGui()
{
    ImGui::Text("Nothing so far");
}

void SparseVoxelOctree::buildOctree(uint1 *positionFragmentList, cudaArray* colorVolumeArray, cudaArray* normalVolumeArray, int fragmentListSize)
{
    m_nodePool.mapToCUDA();
    m_brickPool.mapToCUDA();

    cudaErrorCheck(buildSVO(m_nodePool.getNodePoolDevicePointer(),
             m_nodePool.getNeighbourPoolDevicePointer(),
             m_nodePool.getPoolSize(),
             m_brickPool.getBrickPoolArray(),
             m_brickPool.getResolution(),
             positionFragmentList,
             colorVolumeArray,
             normalVolumeArray,
             fragmentListSize));

    m_brickPool.unmapFromCUDA();
    m_nodePool.unmapFromCUDA();
}

void SparseVoxelOctree::clearOctree()
{
    m_nodePool.mapToCUDA();
    m_nodePool.clearNodePool();
    m_nodePool.unmapFromCUDA();
}
