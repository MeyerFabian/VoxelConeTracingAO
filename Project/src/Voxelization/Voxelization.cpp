#include "Voxelization.h"

#include "externals/GLM/glm/gtc/matrix_transform.hpp"

#include <iostream>
#include "Utilities/errorUtils.h"

extern "C" // this is not necessary imho, but gives a better idea on where the function comes from
{
    cudaError_t setVolumeResulution(int resolution);
}

Voxelization::Voxelization(App *pApp ) :Controllable(pApp, "Voxelisation")
{
    // ### Shader program ###
    mVoxelizationShader = std::unique_ptr<ShaderProgram>(
            new ShaderProgram("/vertex_shaders/voxelization.vert","/fragment_shaders/voxelization.frag", "/geometry_shaders/voxelization.geom"));

    // ### Atomic counter #####
    // Generate atomic buffer
    glGenBuffers(1, &mAtomicBuffer);
    glBindBuffer(GL_ATOMIC_COUNTER_BUFFER, mAtomicBuffer);
    glBufferData(GL_ATOMIC_COUNTER_BUFFER, sizeof(GLuint), NULL, GL_DYNAMIC_DRAW);

    cudaErrorCheck(setVolumeResulution(384)); // todo: gui for cuda..

    resetAtomicCounter();
}

Voxelization::~Voxelization()
{
    // TODO: Delete all the OpenGL stuff
    glDeleteBuffers(1, &mAtomicBuffer);
}


void Voxelization::voxelize(glm::vec3 center, float extent, Scene const * pScene, FragmentList *fragmentList)
{
    // Setup OpenGL for voxelization
    glDisable(GL_DEPTH_TEST);
    glEnable(GL_CULL_FACE);
    glViewport(0, 0, determineVoxeliseResolution(VOXELIZE_RESOLUTION), determineVoxeliseResolution(VOXELIZE_RESOLUTION));

    mVoxelizationShader->use();

    // ### Transformation ###

    // Orthographic projection
    float halfExtent = extent / 2.0f;
    glm::mat4 projection = glm::ortho(
            center.x - halfExtent,
            center.x + halfExtent,
            center.y - halfExtent,
            center.y + halfExtent,
            center.z + halfExtent,
            center.z - halfExtent);
    mVoxelizationShader->updateUniform("model", glm::mat4(1.0));
    mVoxelizationShader->updateUniform("modelNormal", glm::mat4(1.0)); // same since identity
    mVoxelizationShader->updateUniform("projectionView", projection);

    resetAtomicCounter();

    // Bind correct texture slots (TODO: should be done else where and cleaner)
    GLint positionOutputUniformPosition = glGetUniformLocation(static_cast<GLuint>(mVoxelizationShader->getShaderProgramHandle()), "positionOutputImage");
    glUniform1i(positionOutputUniformPosition, 1);
    GLint normalOutputUniformPosition = glGetUniformLocation(static_cast<GLuint>(mVoxelizationShader->getShaderProgramHandle()), "normalOutputImage");
    glUniform1i(normalOutputUniformPosition, 2);
    GLint colorOutputUniformPosition = glGetUniformLocation(static_cast<GLuint>(mVoxelizationShader->getShaderProgramHandle()), "colorOutputImage");
    glUniform1i(colorOutputUniformPosition, 3);

    // Give shader the pixel size for conservative rasterization
    glUniform1f(glGetUniformLocation(static_cast<GLuint>(mVoxelizationShader->getShaderProgramHandle()), "pixelSize"), 2.f / determineVoxeliseResolution(VOXELIZE_RESOLUTION));

    // Bind fragment list with output textures / buffers
    fragmentList->bind();

    // Draw scene with voxelization shader
    for (auto& bucket : pScene->getRenderBuckets())
    {
        // Bind texture of mesh material (pointer to shader is needed for location)
        bucket.first->bind(mVoxelizationShader.get());

        // Draw all meshes in that bucket
        for (Mesh const * pMesh : bucket.second)
        {
            pMesh->draw();
        }
    }

    glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT | GL_ATOMIC_COUNTER_BARRIER_BIT);

    fragmentList->setVoxelCount(readAtomicCounter());

    mVoxelizationShader->disable();
}

GLuint Voxelization::readAtomicCounter() const {// Read atomic counter
    glBindBuffer(GL_ATOMIC_COUNTER_BUFFER, mAtomicBuffer);

    GLuint *mapping = (GLuint*)glMapBufferRange(GL_ATOMIC_COUNTER_BUFFER,
                                                0,
                                                sizeof(GLuint),
                                                GL_MAP_READ_BIT
    );

    glUnmapBuffer(GL_ATOMIC_COUNTER_BUFFER);
    glBindBuffer(GL_ATOMIC_COUNTER_BUFFER, 0);

    return mapping[0];
}

void Voxelization::resetAtomicCounter() const {
    glBindBuffer(GL_ATOMIC_COUNTER_BUFFER, mAtomicBuffer);

    // Map the buffer
    GLuint* mapping = (GLuint*)glMapBufferRange(GL_ATOMIC_COUNTER_BUFFER,
                                                0 ,
                                                sizeof(GLuint),
                                                GL_MAP_WRITE_BIT | GL_MAP_INVALIDATE_BUFFER_BIT | GL_MAP_UNSYNCHRONIZED_BIT
    );
    // Set memory to new value
    memset(mapping, 0, sizeof(GLuint));

    // Unmap the buffer
    glUnmapBuffer(GL_ATOMIC_COUNTER_BUFFER);
    glBindBufferBase(GL_ATOMIC_COUNTER_BUFFER, 0, mAtomicBuffer);

    glBindBuffer(GL_ATOMIC_COUNTER_BUFFER, 0);
}

unsigned int Voxelization::determineVoxeliseResolution(int res) {
    switch (res)
    {
        case VoxelizeResolutions::RES_256 :
            return 256;
        case VoxelizeResolutions::RES_384 :
            return 384;
        case VoxelizeResolutions::RES_512 :
            return 512;
        default:
            return 256;
    }
}

void Voxelization::fillGui()
{
    ImGui::Combo("Resolution",&VOXELIZE_RESOLUTION , " 256x256x256\0 384*384*384\0 512*512*512\0");
}
