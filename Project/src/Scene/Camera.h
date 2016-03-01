#ifndef CAMERA_H_
#define CAMERA_H_

#include "externals/GLM/glm/glm.hpp"
#include "Utilities/enums.h"

class Camera
{
public:

    Camera();
    virtual ~Camera();

    void update(direction dir, float deltaRotationPitch, float deltaRotationYaw);
    void setSpeed(float speed) { mSpeed = speed; }
    glm::mat4 getViewMatrix() const;
    glm::mat4 getProjection(float width, float height) const;
    glm::vec3 getPosition() const;

private:

    // Members
    glm::vec3 mPosition;
    glm::vec3 mDirection;
    float mSpeed;
};

#endif // CAMERA_H_
