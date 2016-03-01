#ifndef CAMERA_H_
#define CAMERA_H_

#include "Utilities/enums.h"
#include "externals/GLM/glm/glm.hpp"

class Camera
{
public:

    Camera();
    virtual ~Camera();

    void update(Direction dir, float deltaRotationPitch, float deltaRotationYaw);
    void setSpeed(float speed) { m_speed = speed; }
    glm::mat4 getViewMatrix() const;
    glm::mat4 getProjection(float width, float height) const;
    glm::vec3 getPosition() const;

private:

    // Members
    glm::vec3 m_position;
    glm::vec3 m_direction;
    float m_speed;
};

#endif // CAMERA_H_
