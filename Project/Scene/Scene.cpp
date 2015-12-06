#include "Scene.h"

#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>

#include "Utilities/errorUtils.h"

Scene::Scene(std::string filepath)
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

    // Iterate over meshes in scene
    for(int i = 0; i < scene->mNumMeshes; i++)
    {
        aiMesh* mesh = scene->mMeshes[i];

        mMeshes.push_back(Mesh(mesh));

        // TODO: Fetch material / sortiere die meshes irgendwie damit: map!

}
