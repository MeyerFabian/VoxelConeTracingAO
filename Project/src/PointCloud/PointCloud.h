#ifndef POINT_CLOUD_H_
#define POINT_CLOUD_H_

#include "Voxelization/FragmentList.h"
#include "Scene/Camera.h"
#include "Rendering/ShaderProgram.h"

#include <memory>

class PointCloud
{
public:

    PointCloud(FragmentList* pFragmentList, Camera const * pCamera);
    virtual ~PointCloud();

    void draw(float width,float height, float volumeExtent);

private:

    FragmentList* mpFragmentList;
    Camera const * mpCamera;
    std::unique_ptr<ShaderProgram> mupShaderProgram;
    GLuint mVAO;
};

#endif // POINT_CLOUD_H_
