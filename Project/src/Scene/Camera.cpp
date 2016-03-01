#include "Camera.h"

#include "externals/GLM/glm/gtc/matrix_transform.hpp"
#include "externals/GLM/glm/gtx/rotate_vector.hpp"

Camera::Camera()
{
    m_position = glm::vec3(-6.45f, 56.7f, -19.4f);
    m_direction = glm::normalize(glm::vec3(-1.0f, -0.5f, 0.5f));
    m_speed = .3f;
}

Camera::~Camera()
{
    // Nothing to do
}

void Camera::update(Direction dir, float deltaRotationYaw, float deltaRotationPitch)
{
    m_direction = glm::rotate(m_direction, deltaRotationYaw, glm::vec3(0,1,0));
    glm::vec3 sideDirection = glm::cross(m_direction, glm::vec3(0,1,0));
    m_direction = glm::rotate(m_direction, deltaRotationPitch, sideDirection);
    switch(dir)
    {
    case FORWARDS:
    {
        m_position += m_speed * m_direction;
        break;
    }
    case LEFT:
    {
        m_position -= m_speed * sideDirection;
        break;
    }
    case BACKWARDS:
    {
        m_position -= m_speed * m_direction;
        break;
    }
    case RIGHT:
    {
        m_position += m_speed * sideDirection;
        break;
    }
    case UP:
    {
        m_position += m_speed * glm::vec3(0,1,0);
        break;
    }
    case DOWN:
    {
        m_position -= m_speed * glm::vec3(0,1,0);
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
    return glm::lookAt(m_position, m_position + m_direction, glm::vec3(0, 1, 0));
}

glm::mat4 Camera::getProjection(float width, float height) const
{
    return glm::perspective(glm::radians(35.0f), width / height, 0.1f, 400.f);
}

glm::vec3 Camera::getPosition() const
{
    return m_position;
}
