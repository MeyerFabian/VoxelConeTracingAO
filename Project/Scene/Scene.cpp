#include "Scene.h"

#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>

#include "Utilities/errorUtils.h"

// TODO: testing
#include <iostream>

Scene::Scene(App* pApp,std::string filepath) : Controllable(pApp, "Scene")
{
    // Create instance of assimp
    Assimp::Importer importer;

    // Import
    const aiScene* scene = importer.ReadFile(filepath,
        aiProcess_CalcTangentSpace       |
        aiProcess_Triangulate            |
        aiProcess_JoinIdenticalVertices  |
        aiProcess_SortByPType);

    // Check whether import was successful
    if(!scene)
    {
        ErrorHandler::log(importer.GetErrorString());
    }

    // Iterate over materials in scene
    for(int i = 0; i < scene->mNumMaterials; i++)
    {
        // Fetch pointer to assimp structure
        aiMaterial* material = scene->mMaterials[i];

        // Create material from assimp data
        std::unique_ptr<Material> upMaterial = std::unique_ptr<Material>(new Material(material));

        // Move to materials
        mMaterials.push_back(std::move(upMaterial));
    }

    // Iterate over meshes in scene
    for(int i = 0; i < scene->mNumMeshes; i++)
    {
        // Fetch pointer to assimp structure
        aiMesh* mesh = scene->mMeshes[i];

        // Create mesh from assimp data
        std::unique_ptr<Mesh> upMesh = std::unique_ptr<Mesh>(new Mesh(mesh));

        // Move to meshes
        mMeshes.push_back(std::move(upMesh));

        // Register in render bucket
        // TODO
    }
}

Scene::~Scene()
{
    // Nothing to do
}

void Scene::fillGui()
{

}
