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

private:

    // Members
    glm::vec3 mPosition;
    glm::vec3 mDirection;
};

#endif // LIGHT_H
