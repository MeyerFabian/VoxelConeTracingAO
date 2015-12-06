#ifndef SCENE_H_
#define SCENE_H_

#include "Mesh.h"
#include "Material.h"
#include <vector>
#include <map>
#include <memory>

class Scene
{
public:

    Scene(std::string filepath);
    virtual ~Scene();

private:

    std::vector<std::unique_ptr<Material> > mMaterials;
    std::vector<std::unique_ptr<Mesh> > mMeshes;
    std::map<Material const *, std::vector<Mesh const *> > mRenderBuckets;
};

#endif // SCENE_H_
