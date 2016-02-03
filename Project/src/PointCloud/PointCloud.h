#ifndef POINT_CLOUD_H_
#define POINT_CLOUD_H_

#include "Voxelization/FragmentList.h"
#include "Scene/Camera.h"
#include "Rendering/ShaderProgram.h"

#include <memory>

class PointCloud
{
public:

    PointCloud(FragmentList* pFragmentList, Camera const * pCamera, GLint pointCount);
    virtual ~PointCloud();

    void draw(float width,float height, glm::vec3 volumeCenter, float volumeExtent);

private:

    FragmentList* mpFragmentList;
    Camera const * mpCamera;
    std::unique_ptr<ShaderProgram> mupShaderProgram;
    GLuint mVAO;
    GLint mPointCount;

};

#endif // POINT_CLOUD_H_
