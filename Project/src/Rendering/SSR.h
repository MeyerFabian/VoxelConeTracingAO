//
// Created by nils1990 on 18.02.16.
//

#ifndef REALTIMERENDERING_SSR_H
#define REALTIMERENDERING_SSR_H

#include <memory>
#include "ShaderProgram.h"
#include "GBuffer.h"
#include "VoxelConeTracing.h"
#include "FullScreenQuad.h"
#include "Cube.h"

class SSR
{
public:
    SSR();
    ~SSR();

    void draw(GBuffer *gbuffer, VoxelConeTracing *vct, glm::mat4 cam, float width, float height);
private:
    std::unique_ptr<ShaderProgram> mSSRShader;

    Cube *mCube; // not actually used as fullscreen quad. just for reflection :D
};


#endif //REALTIMERENDERING_SSR_H
