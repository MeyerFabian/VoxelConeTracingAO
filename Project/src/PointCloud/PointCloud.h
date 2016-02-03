#ifndef POINT_CLOUD_H_
#define POINT_CLOUD_H_

#include "Voxelization/FragmentList.h"
#include "Scene/Camera.h"
#include "Rendering/ShaderProgram.h"

#include <memory>

const int POINT_COUNT = 8000000;

class PointCloud
{
public:

    PointCloud(FragmentList* pFragmentList, Camera const * pCamera);
    virtual ~PointCloud();

    void draw(float width,float height, glm::vec3 volumeCenter, float volumeExtent);

private:

    FragmentList* mpFragmentList;
    Camera const * mpCamera;
    std::unique_ptr<ShaderProgram> mupShaderProgram;
    GLuint mVBO;
    GLuint mVAO;

};

#endif // POINT_CLOUD_H_
