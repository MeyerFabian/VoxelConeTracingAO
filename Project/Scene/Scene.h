#ifndef SCENE_H_
#define SCENE_H_

#include "Mesh.h"
#include <vector>


class Scene
{
public:

    Scene(std::string filepath);
    virtual ~Scene();

private:

    // TODO: Sortiere meshes nach materialien
    std::vector<Mesh> mMeshes;

};

#endif // SCENE_H_
