#ifndef SCENE_H_
#define SCENE_H_

#include "Controllable.h"
#include "Mesh.h"
#include "Material.h"
#include <vector>
#include <map>
#include <memory>

class Scene : public Controllable
{
public:

    Scene(App* pApp, std::string filepath);
    virtual ~Scene();

private:

    virtual void fillGui() override; // Implementation of Controllable

    // Members
    std::vector<std::unique_ptr<Material> > mMaterials;
    std::vector<std::unique_ptr<Mesh> > mMeshes;
    std::map<Material const *, std::vector<Mesh const *> > mRenderBuckets;
};

#endif // SCENE_H_
