#include "Texture.h"

#define STB_IMAGE_IMPLEMENTATION

#include "externals/stb/stb_image.h"
#include "Utilities/errorUtils.h"

#include <vector>

Texture::Texture(std::string filepath)
{
    // Future channel count
    int channelCount;

    // Tell about loading
    ErrorHandler::log(filepath + " is loading.");

    // Flip image at loading
    stbi_set_flip_vertically_on_load(true);

    // Decode image
    unsigned char *data = stbi_load(filepath.c_str(), &m_width, &m_height, &channelCount, 0);

    // Create OpenGL texture
    glGenTextures(1, &m_texture);
    glBindTexture(GL_TEXTURE_2D, m_texture);

    // Repeat texture later
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);

    // Move it to GPU
    if(channelCount == 1)
    {
        glTexImage2D(GL_TEXTURE_2D, 0, GL_R8, m_width, m_height, 0, GL_RED, GL_UNSIGNED_BYTE, data);
    }
    else if(channelCount == 3)
    {
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB8, m_width, m_height, 0, GL_RGB, GL_UNSIGNED_BYTE, data);
    }
    else
    {
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, m_width, m_height, 0, GL_RGBA, GL_UNSIGNED_BYTE, data);
    }

    // Free image data
    stbi_image_free(data);

    // Filtering
    glGenerateMipmap(GL_TEXTURE_2D);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);

    // Unbind texture
    glBindTexture(GL_TEXTURE_2D, 0);
}

Texture::~Texture()
{
    glDeleteTextures(1, &m_texture);
}

void Texture::bind(GLenum slot) const
{
    glActiveTexture(slot);
    glBindTexture(GL_TEXTURE_2D, m_texture);
}
