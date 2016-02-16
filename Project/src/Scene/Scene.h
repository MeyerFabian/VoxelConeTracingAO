#ifndef SCENE_H_
#define SCENE_H_

#include "Controllable.h"
#include "Mesh.h"
#include "Material.h"
#include <Rendering/ShaderProgram.h>
#include "Camera.h"
#include <vector>
#include <map>
#include <memory>
#include "Light.h"
#include "Utilities/enums.h"

// TODO
/*
 * Diffuse has to be used in shader (and deactivated if necessary)
 */

class Scene : public Controllable
{
public:

    Scene(App* pApp, std::string areaName);
    virtual ~Scene();

    void drawDynamicObjectWithCustomShader(ShaderProgram* pShaderProgram) const;

    void updateDynamicObject(glm::vec3 deltaMovement) {mDynamicObjectPosition += deltaMovement;}
    void updateCamera(direction dir, float deltaCameraYaw, float deltaCameraPitch);
    void updateLight(float deltaCameraYaw, float deltaCameraPitch);
    void setCameraSpeed(float speed) { mCamera.setSpeed(speed); }


    glm::vec3 getCamPos() { return mCamera.getPosition(); }

    const std::map<Material const *, std::vector<Mesh const *> >& getRenderBuckets() const{ return mRenderBuckets;}

    const Camera& getCamera() const{ return mCamera;}
    Light& getLight() { return mLight; }

    glm::vec3 getDynamicObjectPosition() const {return mDynamicObjectPosition;}

private:

    virtual void fillGui() override; // Implementation of Controllable

    // Members
    Camera mCamera;
    Light mLight;
    std::vector<std::unique_ptr<Material> > mMaterials;
    std::vector<std::unique_ptr<Mesh> > mMeshes;
    std::map<Material const *, std::vector<Mesh const *> > mRenderBuckets;
    std::unique_ptr<Mesh> mupDynamicMesh;
    std::unique_ptr<Material> mupDynamicMeshMaterial;
    glm::vec3 mDynamicObjectPosition;
};

#endif // SCENE_H_
