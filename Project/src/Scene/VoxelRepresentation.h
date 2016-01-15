//
// Created by miland on 13.01.16.
//

#ifndef REALTIMERENDERING_VOXELREPRESENTATION_H
#define REALTIMERENDERING_VOXELREPRESENTATION_H

#include <GL/gl3w.h>

class VoxelRepresentation {
public:
    VoxelRepresentation();
    void draw(float windowWidth, float windowHeight) const;
private:
    GLuint mTestPointBuffer;
};


#endif //REALTIMERENDERING_VOXELREPRESENTATION_H
