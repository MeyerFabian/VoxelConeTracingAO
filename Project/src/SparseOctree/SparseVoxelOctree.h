#ifndef SPARSE_VOXEL_OCTREE_H
#define SPARSE_VOXEL_OCTREE_H

#include "Controllable.h"
#include "BrickPool.h"
#include "NodePool.h"

class SparseVoxelOctree : public Controllable
{
public:

    // Constructor
    SparseVoxelOctree(App* pApp): Controllable(pApp, "SparseVoxelOctree") {}

    // Destructor
    ~SparseVoxelOctree();

    // Methods
    void init();
    void clearOctree();
    void buildOctree(
        uint1 *positionFragmentList,
        cudaArray* colorVolumeArray,
        cudaArray* normalVolumeArray,
        int fragmentListSize,
        unsigned int voxelizationResolution);
    NodePool& getNodePool() { return m_nodePool; }
    BrickPool& getBrickPool() { return m_brickPool; }

private:

    // Methods
    virtual void fillGui() override; // Implementation of Controllable

    // Members
    BrickPool m_brickPool;
    NodePool m_nodePool;
};


#endif // SPARSE_VOXEL_OCTREE_H
