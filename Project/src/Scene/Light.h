#ifndef LIGHT_H
#define LIGHT_H

#include "externals/GLM/glm/glm.hpp"

class Light{
public:

    Light();
    virtual ~Light();

    void update(float deltaRotationPitch, float deltaRotationYaw);

    glm::mat4 getViewMatrix() const;
    glm::vec3 getPosition() const;
    glm::vec3 getColor() const;
    float getAmbientIntensity() const;
    float getDiffuseIntensity() const;
    void setProjectionMatrix(float width, float height);
    glm::mat4 getProjectionMatrix() const;
    const glm::mat4& getModelMatrix() const;
    float& getAmbientIntensity();
    float& getDiffuseIntensity();

private:

    // Members
    float m_width;
    float m_height;
    glm::vec3 m_position;
    glm::vec3 m_direction;
    float m_ambientIntensity;
    float m_diffuseIntensity;
    glm::vec3 m_color;
    glm::mat4 m_uniformModel;
};

#endif // LIGHT_H
