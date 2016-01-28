#include "Light.h"

#include "externals/GLM/glm/gtc/matrix_transform.hpp"
#include "externals/GLM/glm/gtx/rotate_vector.hpp"

Light::Light()
{
    mPosition = glm::vec3(0,125,0);
	mDirection = glm::vec3(0,-1,0);
}

Light::~Light()
{
    // Nothing to do
}

void Light::update(float movement, float deltaRotationYaw, float deltaRotationPitch)
{
    mDirection = glm::rotate(mDirection, deltaRotationYaw, glm::vec3(1,0,0));
    mDirection = glm::rotate(mDirection, deltaRotationPitch, glm::cross(mDirection, glm::vec3(1,0,0)));
    mPosition += movement * mDirection;

}

glm::mat4 Light::getViewMatrix() const
{
    return glm::lookAt(mPosition, mPosition + mDirection, glm::vec3(1, 0, 0));
}

glm::vec3 Light::getPosition() const
{
    return mPosition;
}
