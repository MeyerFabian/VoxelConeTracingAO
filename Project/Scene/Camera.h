#ifndef CAMERA_H_
#define CAMERA_H_

#include "externals/GLM/glm/glm.hpp"

class Camera
{
public:

    Camera();
    virtual ~Camera();

    void translate(float movement);
    void rotate(glm::vec3 axis, float amount);

    glm::mat4 const * getViewMatrix() const;

private:

    void updateViewMatrix();

    // Members
    glm::mat4 mViewMatrix;
    glm::vec3 mPosition;
    glm::vec3 mDirection;
};

#endif // CAMERA_H_
