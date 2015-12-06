#include <driver_types.h>

extern "C" // this is not necessary imho, but gives a better idea on where the function comes from
{
    void updateBrickPool(cudaArray_t &brickPool);   // hier muss noch der nodepool und die voxelliste hin
    void updateNodePool(cudaArray_t &voxel);        // hier muss noch der nodepool hin
}
