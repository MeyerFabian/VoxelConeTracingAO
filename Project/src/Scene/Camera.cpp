#include "Camera.h"

#include "externals/GLM/glm/gtc/matrix_transform.hpp"
#include "externals/GLM/glm/gtx/rotate_vector.hpp"

Camera::Camera()
{
    mPosition = glm::vec3(-6.45, 56.7, -19.4);
    mDirection = glm::normalize(glm::vec3(-1.0, -0.5, 0.5));
    mSpeed = .3f;
}

Camera::~Camera()
{
    // Nothing to do
}

void Camera::update(direction dir, float deltaRotationYaw, float deltaRotationPitch)
{
    mDirection = glm::rotate(mDirection, deltaRotationYaw, glm::vec3(0,1,0));
    glm::vec3 sideDirection = glm::cross(mDirection, glm::vec3(0,1,0));
    mDirection = glm::rotate(mDirection, deltaRotationPitch, sideDirection);
    switch(dir)
    {
    case FORWARDS:
    {
        mPosition += mSpeed * mDirection;
        break;
    }
    case LEFT:
    {
        mPosition -= mSpeed * sideDirection;
        break;
    }
    case BACKWARDS:
    {
        mPosition -= mSpeed * mDirection;
        break;
    }
    case RIGHT:
    {
        mPosition += mSpeed * sideDirection;
        break;
    }
    case UP:
    {
        mPosition += mSpeed * glm::vec3(0,1,0);
        break;
    }
    case DOWN:
    {
        mPosition -= mSpeed * glm::vec3(0,1,0);
        break;
    }
    case NONE:
    {
        break;
    }


    }
}

glm::mat4 Camera::getViewMatrix() const
{
    return glm::lookAt(mPosition, mPosition + mDirection, glm::vec3(0, 1, 0));
}

glm::mat4 Camera::getProjection(float width, float height) const
{
    return glm::perspective(glm::radians(35.0f), width / height, 0.1f, 400.f);
}

glm::vec3 Camera::getPosition() const
{
    return mPosition;
}
