#ifndef CAMERA_H_
#define CAMERA_H_

#include "externals/GLM/glm/glm.hpp"

class Camera
{
public:

    Camera();
    virtual ~Camera();

    void update(float movement, float deltaRotationPitch, float deltaRotationYaw);

    glm::mat4 getViewMatrix() const;

    glm::vec3 getPosition() const;

private:

    // Members
    glm::vec3 mPosition;
    glm::vec3 mDirection;
};

#endif // CAMERA_H_
