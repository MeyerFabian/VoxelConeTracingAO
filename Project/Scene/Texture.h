#ifndef TEXTURE_H_
#define TEXTURE_H_

#include "externals/picoPNG/picopng.h"
#include <GL/gl3w.h>
#include <string>

class Texture
{
public:

    Texture(std::string filepath);
    virtual ~Texture();

    void bind(GLenum slot) const;

private:

    // Returns count of channels in image
    int loadImage(
        std::string filepath,
        std::vector<unsigned char> &image,
        unsigned long &width,
        unsigned long &height);

    // Members
    unsigned long mWidth;
    unsigned long mHeight;
    GLuint mTexture;

};

#endif // TEXTURE_H_
