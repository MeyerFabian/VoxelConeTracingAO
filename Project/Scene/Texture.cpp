#include "Texture.h"

#include "externals/stb/stb_image.h"
#include "Utilities/errorUtils.h"

#include <vector>

Texture::Texture(std::string filepath)
{
    // Future channel count
    int channelCount;

    // Tell about loading
    ErrorHandler::log(filepath + " is loading.");

    // Decode image
    unsigned char *data = stbi_load(filepath.c_str(), &mWidth, &mHeight, &channelCount, 0);

    // Flip image
    std::vector<unsigned char> flippedImage(mWidth * mHeight * channelCount);

        /* Go over lines */
        for (int i = 0; i < mHeight; i++)
        {
            /* Go over columns */
            for (int j = 0; j < mWidth; j++)
            {
                /* Go over channels */
                for (int k = 0; k < channelCount; k++)
                {
                    flippedImage[i * mWidth * channelCount + j * channelCount + k] = data[(mHeight - 1 - i) * mWidth * channelCount + j * channelCount + k];
                }
            }
        }

    // Create OpenGL texture
    glGenTextures(1, &mTexture);
    glBindTexture(GL_TEXTURE_2D, mTexture);

    // Repeat texture later
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);

    // Move it to GPU
    if(channelCount == 1)
    {
        glTexImage2D(GL_TEXTURE_2D, 0, GL_R8, mWidth, mHeight, 0, GL_RED, GL_UNSIGNED_BYTE, &flippedImage[0]);
    }
    else if(channelCount == 3)
    {
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB8, mWidth, mHeight, 0, GL_RGB, GL_UNSIGNED_BYTE, &flippedImage[0]);
    }
    else
    {
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, mWidth, mHeight, 0, GL_RGBA, GL_UNSIGNED_BYTE, &flippedImage[0]);
    }

    // Free image data
    stbi_image_free(data);

    // Filtering
    glGenerateMipmap(GL_TEXTURE_2D);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR_MIPMAP_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);

    // Unbind texture
    glBindTexture(GL_TEXTURE_2D, 0);
}

Texture::~Texture()
{
    glDeleteTextures(1, &mTexture);
}

void Texture::bind(GLenum slot) const
{
    glActiveTexture(slot);
    glBindTexture(GL_TEXTURE_2D, mTexture);
}
