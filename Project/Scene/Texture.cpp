#include "Texture.h"

#include <fstream>
#include "Utilities/errorUtils.h"

#include <iostream>

Texture::Texture(std::string filepath)
{
    // Read image from disk
    std::vector<unsigned char> image;
    int channelCount = loadImage(filepath, image, mWidth, mHeight);

    // Create OpenGL texture
    glGenTextures(1, &mTexture);
    glBindTexture(GL_TEXTURE_2D, mTexture);

    // Repeat texture later
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);

    // Move it to GPU
    if(channelCount == 1)
    {
        glTexImage2D(GL_TEXTURE_2D, 0, GL_R8, mWidth, mHeight, 0, GL_RED, GL_UNSIGNED_BYTE, &image[0]);
    }
    else if(channelCount = 3)
    {
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB8, mWidth, mHeight, 0, GL_RGB, GL_UNSIGNED_BYTE, &image[0]);
    }
    else
    {
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, mWidth, mHeight, 0, GL_RGBA, GL_UNSIGNED_BYTE, &image[0]);
    }

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

int Texture::loadImage(std::string filepath, std::vector<unsigned char> &image, unsigned long &width, unsigned long &height)
{
    // Read file
    std::ifstream in(filepath.c_str(), std::ios::in | std::ios::binary);

    // Check, whether file was found
    if(in.is_open())
    {
      ErrorHandler::log(filepath + " is loading.");
    }
    else
    {
        ErrorHandler::log(filepath + " was not found. Shit.");
    }

    // Get size
    in.seekg(0, std::ios::end);
    std::streamsize size = in.tellg();

    // Read it
    in.seekg(0, std::ios::beg);
    std::vector<char> buffer(static_cast<unsigned int>(size));
    in.read(&(buffer[0]), static_cast<unsigned int>(size));

    // Close file
    in.close();

    // Decode image
    decodePNG(image, width, height, reinterpret_cast<unsigned char*>(&(buffer[0])), static_cast<unsigned int>(size), false);

    // Calculate number of channels
    unsigned int channelCount = static_cast<unsigned int>(image.size() / (width * height * sizeof(unsigned char)));

    std::cout << channelCount << std::endl;

    // Flip vertical
    std::vector<unsigned char> copyImage(image);

    // Go over lines
    for (unsigned int i = 0; i < height; i++)
    {
        // Go over columns
        for (unsigned int j = 0; j < width; j++)
        {
            // Go over channels
            for (unsigned int k = 0; k < channelCount; k++)
            {
                image[i * width * channelCount + j * channelCount + k] = copyImage[(height - 1 - i) * width * channelCount + j * channelCount + k];
            }
        }
    }
}
