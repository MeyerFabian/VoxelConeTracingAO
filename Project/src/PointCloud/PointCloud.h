/* Visualization of voxel fragment list as point per position. */

#ifndef POINT_CLOUD_H_
#define POINT_CLOUD_H_

#include "src/Voxelization/FragmentList.h"
#include "src/Scene/Camera.h"
#include "src/Rendering/ShaderProgram.h"

#include <memory>

class PointCloud
{
public:

    PointCloud(Camera const * pCamera);
    virtual ~PointCloud();
    void draw(float width, float height, float volumeExtent, FragmentList const * pFragmentList);

private:

    Camera const * m_pCamera;
    std::unique_ptr<ShaderProgram> m_upShaderProgram;
};

#endif // POINT_CLOUD_H_
