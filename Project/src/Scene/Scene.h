#ifndef SCENE_H_
#define SCENE_H_

#include "Controllable.h"
#include "Mesh.h"
#include "Material.h"
#include "Rendering/ShaderProgram.h"
#include "Camera.h"
#include "Light.h"
#include "Utilities/enums.h"

#include <vector>
#include <map>
#include <memory>

class Scene : public Controllable
{
public:

    Scene(App* pApp, std::string areaName);
    virtual ~Scene();

    void draw(ShaderProgram* pShaderProgram, std::string modelUniform) const;

    void updateDynamicObject(glm::vec3 deltaMovement) { m_dynamicObjectPosition += deltaMovement; }
    void updateCamera(Direction dir, float deltaCameraYaw, float deltaCameraPitch);
    void updateLight(float deltaCameraYaw, float deltaCameraPitch);
    void setCameraSpeed(float speed) { m_camera.setSpeed(speed); }
    glm::vec3 getCamPos() { return m_camera.getPosition(); }
    const Camera& getCamera() const{ return m_camera; }
    Light& getLight() { return m_light; }

    virtual void fillGui(); // Implementation of Controllable

private:

    // Members
    Camera m_camera;
    Light m_light;
    std::vector<std::unique_ptr<Material> > m_materials;
    std::vector<std::unique_ptr<Mesh> > m_meshes;
    std::map<Material const *, std::vector<Mesh const *> > m_renderBuckets;
    std::unique_ptr<Mesh> m_upDynamicMesh;
    std::unique_ptr<Material> m_upDynamicMeshMaterial;
    glm::vec3 m_dynamicObjectPosition;
};

#endif // SCENE_H_
