#include "Scene.h"

#include "Utilities/errorUtils.h"
#include "externals/GLM/glm/glm.hpp"
#include "externals/GLM/glm/gtc/matrix_transform.hpp"
#include "externals/GLM/glm/gtx/string_cast.hpp"

#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>

Scene::Scene(App* pApp, std::string areaName) : Controllable(pApp, "Scene")
{
    // Initial members
    m_dynamicObjectPosition = glm::vec3(0,0,0);

    // Create instance of assimp
    Assimp::Importer importer;

    // ### AREA ###

    // Import area
    const aiScene* area = importer.ReadFile(std::string(MESHES_PATH) + "/" + areaName + ".obj",
        aiProcess_GenNormals			 |
        aiProcess_CalcTangentSpace       |
        aiProcess_Triangulate            |
        aiProcess_JoinIdenticalVertices  |
        aiProcess_SortByPType);

    // Check whether import was successful
    if(!area)
    {
        ErrorHandler::log(importer.GetErrorString());
    }

    // Iterate first over materials in area
    for(int i = 0; i < area->mNumMaterials; i++)
    {
        // Fetch pointer to assimp structure
        aiMaterial* material = area->mMaterials[i];

        // Create material from assimp data
        std::unique_ptr<Material> upMaterial = std::unique_ptr<Material>(new Material(areaName, material));

        // Move to materials
        m_materials.push_back(std::move(upMaterial));
    }

    // Iterate then over meshes in area
    for(int i = 0; i < area->mNumMeshes; i++)
    {
        // Fetch pointer to assimp structure
        aiMesh* mesh = area->mMeshes[i];

        // Create mesh from assimp data
        std::unique_ptr<Mesh> upMesh = std::unique_ptr<Mesh>(new Mesh(mesh));

        // Move to meshes
        Mesh const * pMesh = upMesh.get();
        m_meshes.push_back(std::move(upMesh));

        // Register in render bucket
        m_renderBuckets[m_materials[mesh->mMaterialIndex].get()].push_back(pMesh);
    }

    // ### DYNAMIC OBJECT ###

    // Import dynamic object
    const aiScene* dynamicObject = importer.ReadFile(std::string(MESHES_PATH) + "/teapot.obj",
        aiProcess_GenNormals			 |
        aiProcess_CalcTangentSpace       |
        aiProcess_Triangulate            |
        aiProcess_JoinIdenticalVertices  |
        aiProcess_SortByPType);

    // Fetch pointer to one and only material (zero is default material)
    aiMaterial* material = dynamicObject->mMaterials[1];

    // Create material from assimp data
    m_upDynamicMeshMaterial = std::unique_ptr<Material>(new Material("teapot", material));

    // Fetch pointer to assimp structure
    aiMesh* mesh = dynamicObject->mMeshes[0];

    // Create mesh from assimp data
    m_upDynamicMesh = std::unique_ptr<Mesh>(new Mesh(mesh));
}

Scene::~Scene()
{
    // Nothing to do
}

void Scene::draw(ShaderProgram* pShaderProgram, std::string modelUniform) const
{
    // Fill model matrix
    pShaderProgram->updateUniform(modelUniform, glm::mat4(1.0));

    // Draw scene with voxelization shader
    for (auto& bucket : m_renderBuckets)
    {
        // Bind texture of mesh material (pointer to shader is needed for location)
        bucket.first->bind(pShaderProgram);

        // Draw all meshes in that bucket
        for (Mesh const * pMesh : bucket.second)
        {
            pMesh->draw();
        }
    }

    // Set model matrix for dynamic object
    pShaderProgram->updateUniform(modelUniform, glm::translate(glm::mat4(1.0f), m_dynamicObjectPosition));

    // Draw dynamic object
    m_upDynamicMeshMaterial->bind(pShaderProgram);
    m_upDynamicMesh->draw();
}

void Scene::updateCamera(Direction dir, float deltaCameraYaw, float deltaCameraPitch)
{
    // Update camera
    m_camera.update(dir, deltaCameraYaw, deltaCameraPitch);
}

void Scene::updateLight(float deltaCameraYaw, float deltaCameraPitch)
{
    // Update camera
    m_light.update(deltaCameraYaw, deltaCameraPitch);
}

void Scene::fillGui()
{
    std::string output = "Camera " + glm::to_string(m_camera.getPosition());
    ImGui::Text(output.c_str());
    float& ambient = m_light.getAmbientIntensity();
    float& diffuse = m_light.getDiffuseIntensity();
    ImGui::SliderFloat("AmbientIntensity", &ambient, 0.0f, 1.0f, "%0.2f");
    ImGui::SliderFloat("DiffuseIntensity", &diffuse, 0.0f, 2.0f, "%0.2f");
}

