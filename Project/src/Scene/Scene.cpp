#include "Scene.h"

#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>
#include <iostream>
#include "Utilities/errorUtils.h"
#include "externals/GLM/glm/glm.hpp"
#include "externals/GLM/glm/gtc/matrix_transform.hpp"
#include "externals/GLM/glm/gtx/string_cast.hpp"

Scene::Scene(App* pApp, std::string areaName) : Controllable(pApp, "Scene")
{
    // Create instance of assimp
    Assimp::Importer importer;

    // ### AREA ###

    // Import area
    const aiScene* scene = importer.ReadFile(std::string(MESHES_PATH) + "/" + areaName + ".obj",
        aiProcess_GenNormals			 |
        aiProcess_CalcTangentSpace       |
        aiProcess_Triangulate            |
        aiProcess_JoinIdenticalVertices  |
        aiProcess_SortByPType);

    // Check whether import was successful
    if(!scene)
    {
        ErrorHandler::log(importer.GetErrorString());
    }

    // Iterate first over materials in area
    for(int i = 0; i < scene->mNumMaterials; i++)
    {
        // Fetch pointer to assimp structure
        aiMaterial* material = scene->mMaterials[i];

        // Create material from assimp data
        std::unique_ptr<Material> upMaterial = std::unique_ptr<Material>(new Material(areaName, material));

        // Move to materials
        mMaterials.push_back(std::move(upMaterial));
    }

    // Iterate then over meshes in area
    for(int i = 0; i < scene->mNumMeshes; i++)
    {
        // Fetch pointer to assimp structure
        aiMesh* mesh = scene->mMeshes[i];

        // Create mesh from assimp data
        std::unique_ptr<Mesh> upMesh = std::unique_ptr<Mesh>(new Mesh(mesh));

        // Move to meshes
        Mesh const * pMesh = upMesh.get();
        mMeshes.push_back(std::move(upMesh));

        // Register in render bucket
        mRenderBuckets[mMaterials[mesh->mMaterialIndex].get()].push_back(pMesh);
    }

    // ### DYNAMIC OBJECT ###

    // Import dynamic object
    scene = importer.ReadFile(std::string(MESHES_PATH) + "/teapot.obj",
        aiProcess_GenNormals			 |
        aiProcess_CalcTangentSpace       |
        aiProcess_Triangulate            |
        aiProcess_JoinIdenticalVertices  |
        aiProcess_SortByPType);

    // Fetch pointer to one and only material
    aiMaterial* material = scene->mMaterials[0];

    // Create material from assimp data
    mupDynamicMeshMaterial = std::unique_ptr<Material>(new Material("teapot", material));

    // Fetch pointer to assimp structure
    aiMesh* mesh = scene->mMeshes[0];

    // Create mesh from assimp data
    mupDynamicMesh = std::unique_ptr<Mesh>(new Mesh(mesh));
}

Scene::~Scene()
{
    // Nothing to do
}

void Scene::drawDynamicObjectWithCustomShader(ShaderProgram* pShaderProgram) const
{
    mupDynamicMeshMaterial->bind(pShaderProgram);
    mupDynamicMesh->draw();
}

void Scene::updateCamera(direction dir, float deltaCameraYaw, float deltaCameraPitch)
{
    // Update camera
    mCamera.update(dir, deltaCameraYaw, deltaCameraPitch);
}

void Scene::updateLight(float deltaCameraYaw, float deltaCameraPitch)
{
    // Update camera
    mLight.update(deltaCameraYaw, deltaCameraPitch);
}

void Scene::fillGui()
{
    std::string output = "Camera " + glm::to_string(mCamera.getPosition());
    ImGui::Text(output.c_str());
    float& ambient = mLight.getAmbientIntensity();

    float& diffuse = mLight.getDiffuseIntensity();

    ImGui::SliderFloat("AmbientIntensity", &ambient, 0.0f, 1.0f, "%0.2f");
    ImGui::SliderFloat("DiffuseIntensity", &diffuse, 0.0f, 2.0f, "%0.2f");
}

