#include "Light.h"

#include "externals/GLM/glm/gtc/matrix_transform.hpp"
#include "externals/GLM/glm/gtx/rotate_vector.hpp"
#include "externals/GLM/glm/glm.hpp"

Light::Light()
{
    mPosition = glm::vec3(-40,125,0);
	mDirection = glm::vec3(0,-1,0);
	mAmbientIntensity = 0.075f;
	mDiffuseIntensity = 1.3f;
	mColor = glm::vec3(1.0, 0.85, 0.7);
	m_uniformModel = glm::mat4(1.f);
	m_width = 0;
	m_height = 0;
}

Light::~Light()
{
    // Nothing to do
}

void Light::update(float movement, float deltaRotationYaw, float deltaRotationPitch)
{
	float speed = 0.25;
	mDirection = glm::rotate(mDirection, deltaRotationYaw*speed, glm::vec3(1, 0, 0));
	mDirection = glm::rotate(mDirection, deltaRotationPitch*speed, glm::cross(mDirection, glm::vec3(1, 0, 0)));
    

}

glm::mat4 Light::getViewMatrix() const
{
    return glm::lookAt(mPosition, mPosition + mDirection, glm::vec3(1, 0, 0));
}

glm::vec3 Light::getPosition() const
{
    return mPosition;
}
glm::vec3 Light::getColor() const
{
	return mColor;
}
float Light::getAmbientIntensity() const
{
	return mAmbientIntensity;
}
float Light::getDiffuseIntensity() const
{
	return mDiffuseIntensity;
}
glm::mat4 Light::getProjectionMatrix() const
{
	return glm::perspective(glm::radians(35.0f), m_width / m_height, 0.1f, 300.f);
}


const glm::mat4& Light::getModelMatrix() const
{
	return m_uniformModel;
}

void Light::setProjectionMatrix(float width, float height) 
{
	m_width = width;
	m_height = height;
}
