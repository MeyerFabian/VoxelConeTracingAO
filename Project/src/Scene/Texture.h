#ifndef TEXTURE_H_
#define TEXTURE_H_

#include "externals/gl3w/include/GL/gl3w.h"

#include <string>

class Texture
{
public:

    Texture(std::string filepath);
    virtual ~Texture();

    void bind(GLenum slot) const;

private:

    // Members
    int m_width;
    int m_height;
    GLuint m_texture;
};

#endif // TEXTURE_H_
