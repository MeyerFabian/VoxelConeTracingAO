//
// Created by nils1990 on 06.12.15.
//

#include "SparseVoxelOctree.h"

void SparseVoxelOctree::init()
{
    // testsing for node and brick-pool
    m_brickPool.init();
}

void SparseVoxelOctree::fillGui()
{

}

void SparseVoxelOctree::updateOctree()
{
    m_brickPool.registerTextureForCUDAWriting();

    m_brickPool.unregisterTextureForCUDA();
}
