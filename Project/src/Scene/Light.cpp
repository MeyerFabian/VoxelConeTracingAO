#include "Light.h"

#include "externals/GLM/glm/gtc/matrix_transform.hpp"
#include "externals/GLM/glm/gtx/rotate_vector.hpp"
#include "externals/GLM/glm/glm.hpp"

Light::Light()
{
    // Fill members
    m_position = glm::vec3(-40,125,0);
    m_direction = glm::vec3(0,-1,0);
    m_ambientIntensity = 0.4f;
    m_diffuseIntensity = 1.0f;
    m_color = glm::vec3(1.0f, 0.95f, 0.8f);
    m_uniformModel = glm::mat4(1.f);
    m_width = 0;
    m_height = 0;
}

Light::~Light()
{
    // Nothing to do
}

void Light::update(float deltaRotationYaw, float deltaRotationPitch)
{
    float speed = 0.25f;
    m_direction = glm::rotate(m_direction, deltaRotationYaw * speed, glm::vec3(1, 0, 0));
    m_direction = glm::rotate(m_direction, deltaRotationPitch * speed, glm::cross(m_direction, glm::vec3(1, 0, 0)));
}

glm::mat4 Light::getViewMatrix() const
{
    return glm::lookAt(m_position, m_position + m_direction, glm::vec3(1, 0, 0));
}

glm::vec3 Light::getPosition() const
{
    return m_position;
}
glm::vec3 Light::getColor() const
{
    return m_color;
}
float Light::getAmbientIntensity() const
{
    return m_ambientIntensity;
}
float Light::getDiffuseIntensity() const
{
    return m_diffuseIntensity;
}

float& Light::getAmbientIntensity()
{
    return m_ambientIntensity;
}

float& Light::getDiffuseIntensity()
{
    return m_diffuseIntensity;
}

glm::mat4 Light::getProjectionMatrix() const
{
    return glm::perspective(glm::radians(35.0f), m_width / m_height, 0.1f, 150.f);
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
