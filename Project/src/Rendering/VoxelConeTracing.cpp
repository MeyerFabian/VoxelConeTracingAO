#include "VoxelConeTracing.h"

#include "externals/GLM/glm/glm.hpp"
#include "externals/GLM/glm/gtc/matrix_transform.hpp"
#include "externals/GLM/glm/glm.hpp"
#include "externals/GLM/glm/gtc/matrix_transform.hpp"
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
	beginningVoxelSize = 0.05f;
	directionBeginScale = 0.5f;
	maxSteps = 100;
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


	m_gbuffer->init(width, height);

}


void VoxelConeTracing::geometryPass(float width,float height,const std::unique_ptr<Scene>& scene) {
	m_uniformProjection = glm::perspective(glm::radians(35.0f), width / height, 0.1f, 300.f);

	m_gbuffer->onResize(width, height);
    //Bind the GBuffer before enabling (and texture stuff) else it will fail
    m_gbuffer->bindForWriting();

    glm::mat4 uniformModel = glm::mat4(1.f);

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
    m_geomPass->updateUniform("model", uniformModel); // all meshes have center at 0,0,0



    // Render all the buckets' content
    for (auto& bucket : scene->getRenderBuckets())
    {
        // Bind material of bucket (which binds its uniforms and textures)
        bucket.first->bind(m_geomPass.get());

        // Draw all meshes in that bucket
        for (Mesh const * pMesh : bucket.second)
        {
            pMesh->draw();
        }
    }
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

	//Cone Tracing Uniforms
	m_ambientOcclusion->updateUniform("beginningVoxelSize", beginningVoxelSize);
	m_ambientOcclusion->updateUniform("directionBeginScale", directionBeginScale);
	m_ambientOcclusion->updateUniform("maxSteps", maxSteps);
	m_ambientOcclusion->updateUniform("volumeExtent", volumeExtent);
	m_ambientOcclusion->updateUniform("volumeRes", static_cast<float>(brickPool.getResolution().x - 1));


    //Light uniforms
    m_voxelConeTracing->updateUniform("LightPosition", scene->getLight().getPosition());
    m_voxelConeTracing->updateUniform("LightColor", scene->getLight().getColor());
    m_voxelConeTracing->updateUniform("LightAmbientIntensity", scene->getLight().getAmbientIntensity());
    m_voxelConeTracing->updateUniform("LightDiffuseIntensity", scene->getLight().getDiffuseIntensity());

    m_voxelConeTracing->updateUniform("LightModel", scene->getLight().getModelMatrix());
    m_voxelConeTracing->updateUniform("LightView", scene->getLight().getViewMatrix());
    m_voxelConeTracing->updateUniform("LightProjection", scene->getLight().getProjectionMatrix());
    m_voxelConeTracing->updateUniform("shininess", 10.0);
    m_voxelConeTracing->updateUniform("eyeVector", scene->getCamPos());

    //other uniforms
    m_voxelConeTracing->updateUniform("identity", WVP);
	m_voxelConeTracing->updateUniform("screenSize", glm::vec2(width, height));
	m_voxelConeTracing->updateUniform("shadowToWindowRatio", glm::vec2(width / (float)shadowMapResolution, height / (float)shadowMapResolution));

    //GBUFFER TEXTURES
    m_voxelConeTracing->addTexture("positionTex", m_gbuffer->getTextureID(GBuffer::GBUFFER_TEXTURE_TYPE_POSITION));
    m_voxelConeTracing->addTexture("colorTex", m_gbuffer->getTextureID(GBuffer::GBUFFER_TEXTURE_TYPE_DIFFUSE));
    m_voxelConeTracing->addTexture("normalTex", m_gbuffer->getTextureID(GBuffer::GBUFFER_TEXTURE_TYPE_NORMAL));
  
    //LIGHT VIEW MAP TEXTURE
    m_voxelConeTracing->addTexture("LightViewMapTex", lightViewMapTexture);


	GLint brickPoolUniform = glGetUniformLocation(static_cast<GLuint>(m_ambientOcclusion->getShaderProgramHandle()), "brickPool");
	glUniform1i(brickPoolUniform, 2);
	glActiveTexture(GL_TEXTURE2);
	brickPool.bind();


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
	glBlitFramebuffer(0, 0, (GLint)width, (GLint)height, width / 2.0, 0 , width, height / 2.0, GL_COLOR_BUFFER_BIT, GL_LINEAR);

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


}
void VoxelConeTracing::drawAmbientOcclusion(float width, float height, GLuint ScreenQuad, const std::unique_ptr<Scene>& scene, const NodePool& nodePool, const BrickPool& brickPool, const float volumeExtent){
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

	m_ambientOcclusion->use();

	GLint octreeUniform = glGetUniformLocation(static_cast<GLuint>(m_ambientOcclusion->getShaderProgramHandle()), "octree");
	glUniform1i(octreeUniform, 0);
	// bind octree texture
	nodePool.bind();

	//m_ambientOcclusion->updateUniform("eyeVector", scene->getCamPos());

	//other uniforms
	m_ambientOcclusion->updateUniform("identity", WVP);
	m_ambientOcclusion->updateUniform("screenSize", glm::vec2(width, height));
	
	//Cone Tracing Uniforms
	m_ambientOcclusion->updateUniform("beginningVoxelSize", beginningVoxelSize);
	m_ambientOcclusion->updateUniform("directionBeginScale", directionBeginScale);
	m_ambientOcclusion->updateUniform("maxSteps", maxSteps);
	m_ambientOcclusion->updateUniform("volumeExtent", volumeExtent);
	m_ambientOcclusion->updateUniform("volumeRes", static_cast<float>(brickPool.getResolution().x - 1));

	//GBUFFER TEXTURES
	m_ambientOcclusion->addTexture("positionTex", m_gbuffer->getTextureID(GBuffer::GBUFFER_TEXTURE_TYPE_POSITION));
	m_ambientOcclusion->addTexture("normalTex", m_gbuffer->getTextureID(GBuffer::GBUFFER_TEXTURE_TYPE_NORMAL));    m_ambientOcclusion->updateUniform("volumeRes", static_cast<float>(brickPool.getResolution().x - 1));
	
	GLint brickPoolUniform = glGetUniformLocation(static_cast<GLuint>(m_ambientOcclusion->getShaderProgramHandle()), "brickPool");
	glUniform1i(brickPoolUniform, 2);
	glActiveTexture(GL_TEXTURE2);
	brickPool.bind();
	
	//Draw FullScreenQuad
	glBindVertexArray(ScreenQuad);
	glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
	glBindVertexArray(0);

	m_ambientOcclusion->disable();

}

void VoxelConeTracing::fillGui(){
	ImGui::SliderFloat("beginning voxel size", &beginningVoxelSize, 0.01f, 1.0f, "%.3f");
	ImGui::SliderInt("max steps cone tracing", &maxSteps, 50, 2000, "%.0f");
	ImGui::SliderFloat("position begin", &directionBeginScale, 0.0f, 5.0f, "%.1f");
}