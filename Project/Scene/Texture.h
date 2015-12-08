#ifndef TEXTURE_H_
#define TEXTURE_H_

#include <GL/gl3w.h>
#include <string>

class Texture
{
public:

    Texture(std::string filepath);
    virtual ~Texture();

    void bind(GLenum slot) const;

private:

    // Members
    int mWidth;
    int mHeight;
    GLuint mTexture;
};

#endif // TEXTURE_H_
