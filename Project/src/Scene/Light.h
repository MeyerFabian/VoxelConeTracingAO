#ifndef LIGHT_H
#define LIGHT_H

#include "externals/GLM/glm/glm.hpp"

class Light
{
public:

	Light();
	virtual ~Light();

    void update(float movement, float deltaRotationPitch, float deltaRotationYaw);

    glm::mat4 getViewMatrix() const;

    glm::vec3 getPosition() const;
	glm::vec3 getColor() const;
	float getAmbientIntensity() const;
	float getDiffuseIntensity() const;
	void setProjectionMatrix(float width, float height);
	glm::mat4 getProjectionMatrix() const;
	const glm::mat4& getModelMatrix() const;
private:
	float m_width;
	float m_height;
    // Members
    glm::vec3 mPosition;
    glm::vec3 mDirection;
	float mAmbientIntensity;
	float mDiffuseIntensity;
	glm::vec3 mColor;
	glm::mat4 m_uniformModel;
};

#endif // LIGHT_H
