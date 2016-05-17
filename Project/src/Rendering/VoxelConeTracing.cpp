#include "VoxelConeTracing.h"

#include "externals/GLM/glm/glm.hpp"
#include "externals/GLM/glm/gtc/matrix_transform.hpp"
#include "externals/GLM/glm/glm.hpp"
#include "externals/GLM/glm/gtx/string_cast.hpp"
#include <iostream>
using namespace std;

#ifdef __unix__
template<typename T, typename... Args>
std::unique_ptr<T> make_unique(Args&&... args) {
    return std::unique_ptr<T>(new T(std::forward<Args>(args)...));
}
#endif

using namespace std;

VoxelConeTracing::VoxelConeTracing(App* pApp) : Controllable(pApp, "Voxel Cone Tracing")
{
    m_gbuffer = make_unique<GBuffer>();
    directionBeginScale = 3.0f;
    ambientOcclusionScale = 0.25f;
    maxDistance = 5.0f;
    lambda = 1.0f;
    colorBleeding = 0.0f;

    glGenFramebuffers(1, &mPhongFbo);
}


VoxelConeTracing::~VoxelConeTracing()
{
}
void VoxelConeTracing::init(float width,float height) {
    // Prepare the one and only shader
    m_geomPass = make_unique<ShaderProgram>("/vertex_shaders/geom_pass.vert", "/fragment_shaders/geom_pass.frag");
    m_voxelConeTracing = make_unique<ShaderProgram>("/vertex_shaders/voxelConeTracing.vert", "/fragment_shaders/voxelConeTracing.frag");
    m_ambientOcclusion= make_unique<ShaderProgram>("/vertex_shaders/voxelConeTracing.vert", "/fragment_shaders/ambientOcclusion.frag");
    m_phongShading = make_unique<ShaderProgram>("/vertex_shaders/voxelConeTracing.vert", "/fragment_shaders/phong.frag");
    m_voxelGlow = make_unique<ShaderProgram>("/vertex_shaders/voxelConeTracing.vert", "/fragment_shaders/voxelGlow.frag");

    m_gbuffer->init(width, height);

    glBindFramebuffer(GL_DRAW_FRAMEBUFFER, mPhongFbo);
    glGenTextures(1, &mPhongTexture);
    glBindTexture(GL_TEXTURE_2D, mPhongTexture);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, width, height, 0, GL_RGB, GL_FLOAT, NULL);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glFramebufferTexture2D(GL_DRAW_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, mPhongTexture, 0);

    GLenum DrawBuffers[] = { GL_COLOR_ATTACHMENT0};
    glDrawBuffers(1, DrawBuffers);

    GLenum Status = glCheckFramebufferStatus(GL_FRAMEBUFFER);
    if (Status != GL_FRAMEBUFFER_COMPLETE)
    {
        printf("FB error: 0x%x\n", Status);
    }
    else
    {
        printf("GBuffer successfully initialized.\n");
    }
    glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0);
}


void VoxelConeTracing::geometryPass(float width,float height,const std::unique_ptr<Scene>& scene) {
    m_uniformProjection = glm::perspective(glm::radians(35.0f), width / height, 0.1f, 300.f);

    m_gbuffer->onResize(width, height);
    //Bind the GBuffer before enabling (and texture stuff) else it will fail
    m_gbuffer->bindForWriting();

    glDepthMask(GL_TRUE);
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_CULL_FACE);
    glDisable(GL_BLEND);

    // Use the one and only shader
    m_geomPass->use();

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    // Create uniforms used by shader
    // Fill uniforms to shader
    m_geomPass->updateUniform("projection", m_uniformProjection);
    m_geomPass->updateUniform("view", scene->getCamera().getViewMatrix());

    // Draw with custom shader
    scene->draw(m_geomPass.get(), "model");

    m_geomPass->disable();

    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glDepthMask(GL_FALSE);
}

void VoxelConeTracing::drawSimplePhong(float width, float height,
    int shadowMapResolution, GLuint ScreenQuad,
    const GLuint lightViewMapTexture,
    const std::unique_ptr<Scene>& scene) const

{
    glDisable(GL_DEPTH_TEST);

    //Bind window framebuffer
    glViewport(0, 0, width, height);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    //Bind Gbuffer so we can transfer the geometry information into the color coded main framebuffer
    m_gbuffer->bindForReading();
    glBindFramebuffer(GL_DRAW_FRAMEBUFFER, mPhongFbo);


    glm::mat4 WVP = glm::mat4(1.f);

    m_phongShading->use();

    //Light uniforms
    m_phongShading->updateUniform("LightPosition", scene->getLight().getPosition());
    m_phongShading->updateUniform("LightColor", scene->getLight().getColor());
    m_phongShading->updateUniform("LightAmbientIntensity", scene->getLight().getAmbientIntensity());
    m_phongShading->updateUniform("LightDiffuseIntensity", scene->getLight().getDiffuseIntensity());

    m_phongShading->updateUniform("LightModel", scene->getLight().getModelMatrix());
    m_phongShading->updateUniform("LightView", scene->getLight().getViewMatrix());
    m_phongShading->updateUniform("LightProjection", scene->getLight().getProjectionMatrix());
    //Specular
    //m_phongShading->updateUniform("shininess", 10.0);
    //m_phongShading->updateUniform("eyeVector", scene->getCamPos());

    //other uniforms
    m_phongShading->updateUniform("identity", WVP);
    m_phongShading->updateUniform("screenSize", glm::vec2(width, height));
    m_phongShading->updateUniform("shadowToWindowRatio", glm::vec2(width / (float)shadowMapResolution, height / (float)shadowMapResolution));

    //GBUFFER TEXTURES
    m_phongShading->addTexture("positionTex", m_gbuffer->getTextureID(GBuffer::GBUFFER_TEXTURE_TYPE_POSITION));
    m_phongShading->addTexture("colorTex", m_gbuffer->getTextureID(GBuffer::GBUFFER_TEXTURE_TYPE_DIFFUSE));
    m_phongShading->addTexture("normalTex", m_gbuffer->getTextureID(GBuffer::GBUFFER_TEXTURE_TYPE_NORMAL));

    //LIGHT VIEW MAP TEXTURE
    m_phongShading->addTexture("LightViewMapTex", lightViewMapTexture);



    //Draw FullScreenQuad  DRAW TWICE ONE TO FBO AND ONE FULLSCREEN.. it might be faster to just display the fbo, but im too lazy :D
    glBindVertexArray(ScreenQuad);
    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
    glBindVertexArray(0);

    glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0);

    //Draw FullScreenQuad
    glBindVertexArray(ScreenQuad);
    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
    glBindVertexArray(0);

    m_phongShading->disable();

}

void VoxelConeTracing::drawVoxelConeTracing(float width, float height,
                            int shadowMapResolution, GLuint ScreenQuad,
                            const GLuint lightViewMapTexture,
                            const std::unique_ptr<Scene>& scene,
                            const NodePool& nodePool,
                            BrickPool& brickPool, const float stepSize, const float volumeExtent) const

{
    glDisable(GL_DEPTH_TEST);

    //Bind window framebuffer
    glViewport(0, 0, width, height);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    //Bind Gbuffer so we can transfer the geometry information into the color coded main framebuffer

    m_gbuffer->bindForReading();

    glm::mat4 WVP = glm::mat4(1.f);

    m_voxelConeTracing->use();

    GLint octreeUniform = glGetUniformLocation(static_cast<GLuint>(m_ambientOcclusion->getShaderProgramHandle()), "octree");
    glUniform1i(octreeUniform, 0);
    // bind octree texture
    nodePool.bind();

    GLint brickPoolUniform = glGetUniformLocation(static_cast<GLuint>(m_ambientOcclusion->getShaderProgramHandle()), "brickPool");
    glUniform1i(brickPoolUniform, 5);
    glActiveTexture(GL_TEXTURE5);
    brickPool.bind();


    //Cone Tracing Uniforms
    //m_voxelConeTracing->updateUniform("beginningVoxelSize", beginningVoxelSize);
    m_voxelConeTracing->updateUniform("directionBeginScale", directionBeginScale);
    m_voxelConeTracing->updateUniform("maxDistance", maxDistance);
    m_voxelConeTracing->updateUniform("volumeExtent", volumeExtent);
    m_voxelConeTracing->updateUniform("volumeRes", static_cast<float>(brickPool.getResolution().x));
    m_voxelConeTracing->updateUniform("lambda", lambda);
    m_voxelConeTracing->updateUniform("ambientOcclusionScale", ambientOcclusionScale);
    m_voxelConeTracing->updateUniform("colorBleeding", colorBleeding);
    //Light uniforms
    m_voxelConeTracing->updateUniform("LightPosition", scene->getLight().getPosition());
    m_voxelConeTracing->updateUniform("LightColor", scene->getLight().getColor());
    m_voxelConeTracing->updateUniform("LightAmbientIntensity", scene->getLight().getAmbientIntensity());
    m_voxelConeTracing->updateUniform("LightDiffuseIntensity", scene->getLight().getDiffuseIntensity());

    m_voxelConeTracing->updateUniform("LightModel", scene->getLight().getModelMatrix());
    m_voxelConeTracing->updateUniform("LightView", scene->getLight().getViewMatrix());
    m_voxelConeTracing->updateUniform("LightProjection", scene->getLight().getProjectionMatrix());
    //m_voxelConeTracing->updateUniform("shininess", 10.0);
    // m_voxelConeTracing->updateUniform("eyeVector", scene->getCamPos());

    //other uniforms
    m_voxelConeTracing->updateUniform("identity", WVP);
    m_voxelConeTracing->updateUniform("screenSize", glm::vec2(width, height));
    m_voxelConeTracing->updateUniform("shadowToWindowRatio", glm::vec2(width / (float)shadowMapResolution, height / (float)shadowMapResolution));

    //GBUFFER TEXTURES
    m_voxelConeTracing->addTexture("positionTex", m_gbuffer->getTextureID(GBuffer::GBUFFER_TEXTURE_TYPE_POSITION));
    m_voxelConeTracing->addTexture("colorTex", m_gbuffer->getTextureID(GBuffer::GBUFFER_TEXTURE_TYPE_DIFFUSE));
    m_voxelConeTracing->addTexture("normalTex", m_gbuffer->getTextureID(GBuffer::GBUFFER_TEXTURE_TYPE_NORMAL));
    m_voxelConeTracing->addTexture("tangentTex", m_gbuffer->getTextureID(GBuffer::GBUFFER_TEXTURE_TYPE_TANGENT));

    //LIGHT VIEW MAP TEXTURE
    m_voxelConeTracing->addTexture("LightViewMapTex", lightViewMapTexture);




    //Draw FullScreenQuad
    glBindVertexArray(ScreenQuad);
    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
    glBindVertexArray(0);

    m_voxelConeTracing->disable();

}
void VoxelConeTracing::drawGBuffer(float width, float height){
    //Bind window framebuffer

    glDisable(GL_DEPTH_TEST);
    glViewport(0, 0, width, height);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    m_gbuffer->bindForReading();

    m_gbuffer->setReadBuffer(GBuffer::GBUFFER_TEXTURE_TYPE_POSITION);
    glBlitFramebuffer(0, 0, (GLint)width, (GLint)height, 0, height / 2.0, width / 2.0, height, GL_COLOR_BUFFER_BIT, GL_LINEAR);

    m_gbuffer->setReadBuffer(GBuffer::GBUFFER_TEXTURE_TYPE_DIFFUSE);
    glBlitFramebuffer(0, 0, (GLint)width, (GLint)height, width / 2.0, height / 2.0, width, height, GL_COLOR_BUFFER_BIT, GL_LINEAR);

    m_gbuffer->setReadBuffer(GBuffer::GBUFFER_TEXTURE_TYPE_NORMAL);
    glBlitFramebuffer(0, 0, (GLint)width, (GLint)height, 0, 0 , width/2.0, height / 2.0, GL_COLOR_BUFFER_BIT, GL_LINEAR);

    m_gbuffer->setReadBuffer(GBuffer::GBUFFER_TEXTURE_TYPE_TANGENT);
    glBlitFramebuffer(0, 0, (GLint)width, (GLint)height, width / 2.0, 0, width, height / 2.0, GL_COLOR_BUFFER_BIT, GL_LINEAR);

}


void VoxelConeTracing::drawGBufferPanels(float width, float height){
    //Bind window framebuffer

    glDisable(GL_DEPTH_TEST);
    glViewport(0, 0, width, height);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    m_gbuffer->bindForReading();

    m_gbuffer->setReadBuffer(GBuffer::GBUFFER_TEXTURE_TYPE_POSITION);
    glBlitFramebuffer(0, 0, (GLint)width, (GLint)height, 150, 0, 300, 150, GL_COLOR_BUFFER_BIT, GL_LINEAR);

    m_gbuffer->setReadBuffer(GBuffer::GBUFFER_TEXTURE_TYPE_DIFFUSE);
    glBlitFramebuffer(0, 0, (GLint)width, (GLint)height, 300, 0, 450, 150, GL_COLOR_BUFFER_BIT, GL_LINEAR);

    m_gbuffer->setReadBuffer(GBuffer::GBUFFER_TEXTURE_TYPE_NORMAL);
    glBlitFramebuffer(0, 0, (GLint)width, (GLint)height, 450, 0, 600, 150, GL_COLOR_BUFFER_BIT, GL_LINEAR);

    m_gbuffer->setReadBuffer(GBuffer::GBUFFER_TEXTURE_TYPE_TANGENT);
    glBlitFramebuffer(0, 0, (GLint)width, (GLint)height, 600, 0, 750, 150, GL_COLOR_BUFFER_BIT, GL_LINEAR);

}

void VoxelConeTracing::drawAmbientOcclusion(float width, float height, GLuint ScreenQuad, const std::unique_ptr<Scene>& scene, const NodePool& nodePool, const BrickPool& brickPool, const float volumeExtent){
    //Bind window framebuffer

    glDisable(GL_DEPTH_TEST);
    glViewport(0, 0, width, height);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    m_gbuffer->bindForReading();

    glm::mat4 WVP = glm::mat4(1.f);

    m_ambientOcclusion->use();

    GLint octreeUniform = glGetUniformLocation(static_cast<GLuint>(m_ambientOcclusion->getShaderProgramHandle()), "octree");
    glUniform1i(octreeUniform, 0);
    // bind octree texture
    nodePool.bind();

    GLint brickPoolUniform = glGetUniformLocation(static_cast<GLuint>(m_ambientOcclusion->getShaderProgramHandle()), "brickPool");
    glUniform1i(brickPoolUniform, 5);
    glActiveTexture(GL_TEXTURE5);
    brickPool.bind();
    //m_ambientOcclusion->updateUniform("eyeVector", scene->getCamPos());

    //other uniforms
    m_ambientOcclusion->updateUniform("identity", WVP);
    m_ambientOcclusion->updateUniform("screenSize", glm::vec2(width, height));


    //other uniforms
    m_ambientOcclusion->updateUniform("identity", WVP);
    m_ambientOcclusion->updateUniform("screenSize", glm::vec2(width, height));

    //Cone Tracing Uniforms
    m_ambientOcclusion->updateUniform("beginningVoxelSize", beginningVoxelSize);
    m_ambientOcclusion->updateUniform("directionBeginScale", directionBeginScale);
    m_ambientOcclusion->updateUniform("maxDistance", maxDistance);
    m_ambientOcclusion->updateUniform("volumeExtent", volumeExtent);
    m_ambientOcclusion->updateUniform("volumeRes", static_cast<float>(brickPool.getResolution().x ));
    m_ambientOcclusion->updateUniform("lambda", lambda);



    //GBUFFER TEXTURES
    m_ambientOcclusion->addTexture("positionTex", m_gbuffer->getTextureID(GBuffer::GBUFFER_TEXTURE_TYPE_POSITION));
    m_ambientOcclusion->addTexture("normalTex", m_gbuffer->getTextureID(GBuffer::GBUFFER_TEXTURE_TYPE_NORMAL));
    m_ambientOcclusion->addTexture("tangentTex", m_gbuffer->getTextureID(GBuffer::GBUFFER_TEXTURE_TYPE_TANGENT));



    //Draw FullScreenQuad
    glBindVertexArray(ScreenQuad);
    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
    glBindVertexArray(0);

    m_ambientOcclusion->disable();

}

void VoxelConeTracing::drawVoxelGlow(float width, float height, GLuint ScreenQuad, const std::unique_ptr<Scene>& scene, const NodePool& nodePool, const BrickPool& brickPool, const float volumeExtent){
    //Bind window framebuffer

    glDisable(GL_DEPTH_TEST);
    glViewport(0, 0, width, height);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    //glEnable(GL_BLEND);
    // glBlendEquation(GL_FUNC_ADD);
    // glBlendFunc(GL_ONE, GL_ONE); //BLEND_FUNCTION BY OPENGL MAY USE (GL_SRC_ALPHA/GL_ONE_MINUS_SRC_ALPHA) for transparency

    m_gbuffer->bindForReading();

    //Bind Gbuffer so we can transfer the geometry information into the color coded main framebuffer
    //glClear(GL_COLOR_BUFFER_BIT);


    glm::mat4 WVP = glm::mat4(1.f);

    m_voxelGlow->use();

    GLint octreeUniform = glGetUniformLocation(static_cast<GLuint>(m_voxelGlow->getShaderProgramHandle()), "octree");
    glUniform1i(octreeUniform, 0);
    // bind octree texture
    nodePool.bind();

    GLint brickPoolUniform = glGetUniformLocation(static_cast<GLuint>(m_voxelGlow->getShaderProgramHandle()), "brickPool");
    glUniform1i(brickPoolUniform, 5);
    glActiveTexture(GL_TEXTURE5);
    brickPool.bind();
    //m_ambientOcclusion->updateUniform("eyeVector", scene->getCamPos());

    //other uniforms
    m_voxelGlow->updateUniform("identity", WVP);
    m_voxelGlow->updateUniform("screenSize", glm::vec2(width, height));

    //Cone Tracing Uniforms
    m_voxelGlow->updateUniform("beginningVoxelSize", beginningVoxelSize);
    m_voxelGlow->updateUniform("directionBeginScale", directionBeginScale);
    m_voxelGlow->updateUniform("maxDistance", maxDistance);
    m_voxelGlow->updateUniform("volumeExtent", volumeExtent);
    m_voxelGlow->updateUniform("volumeRes", static_cast<float>(brickPool.getResolution().x));

    //GBUFFER TEXTURES
    m_voxelGlow->addTexture("positionTex", m_gbuffer->getTextureID(GBuffer::GBUFFER_TEXTURE_TYPE_POSITION));
    m_voxelGlow->addTexture("normalTex", m_gbuffer->getTextureID(GBuffer::GBUFFER_TEXTURE_TYPE_NORMAL));
    m_voxelGlow->addTexture("tangentTex", m_gbuffer->getTextureID(GBuffer::GBUFFER_TEXTURE_TYPE_TANGENT));



    //Draw FullScreenQuad
    glBindVertexArray(ScreenQuad);
    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
    glBindVertexArray(0);

    m_voxelGlow->disable();

}
void VoxelConeTracing::fillGui(){
    ImGui::SliderFloat("max distance", &maxDistance, 0.5f, 20.0f , "%.2f");
    ImGui::SliderFloat("Pushed out PerimeterStart in VoxelSize", &directionBeginScale, 0.0f, 5.0f, "%.1f");
    ImGui::SliderFloat("AO lambda", &lambda, 0.0f, 2.0f, "%.3f");
    ImGui::SliderFloat("AO scale VoxelConeTracing", &ambientOcclusionScale, 0.0f, 1.0f, "%.3f");
    ImGui::SliderFloat("Color Bleeding", &colorBleeding, 0.0f, 5.0f, "%.2f");

}

GLuint VoxelConeTracing::getPhongTexID() {
    return mPhongTexture;
}
