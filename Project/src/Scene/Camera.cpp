#include "Camera.h"

#include "externals/GLM/glm/gtc/matrix_transform.hpp"
#include "externals/GLM/glm/gtx/rotate_vector.hpp"

Camera::Camera()
{
    mPosition = glm::vec3(0,0,0);
    mDirection = glm::vec3(0,0,-1);
}

Camera::~Camera()
{
    // Nothing to do
}

void Camera::update(float movement, float deltaRotationYaw, float deltaRotationPitch)
{
    mDirection = glm::rotate(mDirection, deltaRotationYaw, glm::vec3(0,1,0));
    mDirection = glm::rotate(mDirection, deltaRotationPitch, glm::cross(mDirection, glm::vec3(0,1,0)));
    mPosition += movement * mDirection;

}

glm::mat4 Camera::getViewMatrix() const
{
    return glm::lookAt(mPosition, mPosition + mDirection, glm::vec3(0, 1, 0));
}

glm::vec3 Camera::getPosition() const
{
    return mPosition;
}
