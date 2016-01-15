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

// TODO
/*
 * Diffuse has to be used in shader (and deactivated if necessary)
 */

class Scene : public Controllable
{
public:

    Scene(App* pApp, std::string filepath);
    virtual ~Scene();

    void update(float movement, float deltaCameraYaw, float deltaCameraPitch);
    void draw(float windowWidth,float windowHeight) const;

    void drawWithCustomShader() const;

private:

    virtual void fillGui() override; // Implementation of Controllable

    // Members
    Camera mCamera;
    std::unique_ptr<ShaderProgram> mupShader;
    std::vector<std::unique_ptr<Material> > mMaterials;
    std::vector<std::unique_ptr<Mesh> > mMeshes;
    std::map<Material const *, std::vector<Mesh const *> > mRenderBuckets;
};

#endif // SCENE_H_
