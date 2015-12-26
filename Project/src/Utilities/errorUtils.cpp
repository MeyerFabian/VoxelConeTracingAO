//
// Created by nils1990 on 04.12.15.
//

#include "errorUtils.h"
#include "CudaException.h"

#include <iostream>

void ErrorHandler::gpuAssert(cudaError_t code, const char *file, int line)
{
    if (code != cudaSuccess)
    {
        throw(CudaException(std::string("CUDA Error: ") + cudaGetErrorString(code) + " " + file + " " + std::to_string(line)));
    }
}

GLenum ErrorHandler::checkGLError(bool printIfNoError)
{
    GLenum error = glGetError();

    switch (error)
    {
        case GL_INVALID_VALUE:
        std::cerr << "GL_INVALID_VALUE" << std::endl;
            break;
        case GL_INVALID_ENUM:
            std::cerr << "GL_INVALID_ENUM" << std::endl;
            break;
        case GL_INVALID_OPERATION:
            std::cerr << "GL_INVALID_OPERATION" << std::endl;
            break;
        case GL_INVALID_FRAMEBUFFER_OPERATION:
            std::cerr << "GL_INVALID_FRAMEBUFFER_OPERATION" << std::endl;
            break;
        case GL_OUT_OF_MEMORY:
            std::cerr << "GL_OUT_OF_MEMORY" << std::endl;
            break;
        case GL_NO_ERROR:
            if (printIfNoError)
            {
                std::cerr << "GL_NO_ERROR" << std::endl;
            }
            break;
    }
    return error;
}

void ErrorHandler::log(std::string message)
{
    std::cout << message << std::endl;
}
