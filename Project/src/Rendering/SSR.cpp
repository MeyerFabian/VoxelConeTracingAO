//
// Created by nils1990 on 18.02.16.
//

#include "externals/GLM/glm/glm.hpp"
#include "externals/GLM/glm/gtc/matrix_transform.hpp"
#include "externals/GLM/glm/glm.hpp"
#include "externals/GLM/glm/gtx/string_cast.hpp"

#include "SSR.h"

#ifdef __unix__
template<typename T, typename... Args>
std::unique_ptr<T> std::make_unique(Args&&... args) {
    return std::unique_ptr<T>(new T(std::forward<Args>(args)...));
}
#endif

SSR::SSR()
{
    mSSRShader = std::make_unique<ShaderProgram>("/vertex_shaders/SSR.vert", "/fragment_shaders/SSR.frag");
    mCube = new Cube(10);
}

SSR::~SSR()
{
    delete mCube;
}

void SSR::draw(GBuffer *gbuffer, VoxelConeTracing *vct, glm::mat4 cam, float width, float height)
{
    glEnable(GL_DEPTH_TEST);
    glm::mat4 projection = glm::perspective(glm::radians(35.0f), width / height, 0.1f, 300.f);
    gbuffer->getDepthTextureID();
    vct->getPhongTexID();

    mSSRShader->updateUniform("projection", projection);
    mSSRShader->updateUniform("view",cam);
    mSSRShader->updateUniform("model", glm::mat4(1.f)); // all meshes have center at 0,0,0

    mSSRShader->use();
    mCube->render();
    mSSRShader->disable();
    glDisable(GL_DEPTH_TEST);
}
