#include "Camera.h"

#include "externals/GLM/glm/gtc/matrix_transform.hpp"
#include "externals/GLM/glm/gtx/rotate_vector.hpp"

Camera::Camera()
{
    mPosition = glm::vec3(0,50,50);
    mDirection = glm::vec3(0,0,-1);
}

Camera::~Camera()
{
    // Nothing to do
}

void Camera::translate(float movement)
{
    mPosition += movement * mDirection;
}

void Camera::rotate(glm::vec3 axis, float amount)
{
    mDirection = glm::rotate(mDirection, amount, axis);

}

glm::mat4 Camera::getViewMatrix() const
{
    return glm::lookAt(mPosition, mPosition + mDirection, glm::vec3(0, 1, 0));
}
