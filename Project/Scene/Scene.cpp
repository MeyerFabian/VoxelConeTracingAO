#include "Scene.h"

#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>

#include "Utilities/errorUtils.h"
#include "externals/GLM/glm/glm.hpp"
#include "externals/GLM/glm/gtc/matrix_transform.hpp"

Scene::Scene(App* pApp,std::string filepath) : Controllable(pApp, "Scene")
{
    // Prepare the one and only shader
    mupShader = std::unique_ptr<ShaderProgram>(new ShaderProgram("/vertex_shaders/sponza.vert","/fragment_shaders/sponza.frag"));

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

    // Iterate first over materials in scene
    for(int i = 0; i < scene->mNumMaterials; i++)
    {
        // Fetch pointer to assimp structure
        aiMaterial* material = scene->mMaterials[i];

        // Create material from assimp data
        std::unique_ptr<Material> upMaterial = std::unique_ptr<Material>(new Material(material));

        // Move to materials
        mMaterials.push_back(std::move(upMaterial));
    }

    // Iterate then over meshes in scene
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
}

Scene::~Scene()
{
    // Nothing to do
}

void Scene::draw() const
{
    // Use the one and only shader
    mupShader->use();

    // TODO: TEST
    glm::mat4 uniformView = glm::lookAt(glm::vec3(0, 0, -10),glm::vec3(0, 0, 1), glm::vec3(0, 1, 0));
    glm::mat4 uniformProjection = glm::perspective(glm::radians(35.0f), ((GLfloat)800 / (GLfloat)600), 0.1f, 100.f);
    glm::mat4 uniformModel = glm::mat4(1.f);

    mupShader->updateUniform("color", glm::vec4(1,1,1,1));
    mupShader->updateUniform("projection", uniformProjection);
    mupShader->updateUniform("view", uniformView);
    mupShader->updateUniform("model", uniformModel); // all meshes have center at 0,0,0

    // Render all the buckets' content
    for(auto& bucket : mRenderBuckets)
    {
        // Bind material of bucket (which binds its uniforms and textures)
        bucket.first->bind(mupShader.get());

        // Draw all meshes in that bucket
        for(Mesh const * pMesh : bucket.second)
        {
            pMesh->draw();
        }
    }
}

void Scene::fillGui()
{

}
