#ifndef ERROR_UTILS_H
#define ERROR_UTILS_H

#include "externals/gl3w/include/GL/gl3w.h"
#include "externals/GLFW/include/GLFW/glfw3.h"

#include <cuda_runtime.h>
#include <string>

#define cudaErrorCheck(ans) { ErrorHandler::gpuAssert((ans), __FILE__, __LINE__); }

class ErrorHandler
{
public:

    static void gpuAssert(cudaError_t code, const char *file, int line);

    static GLenum checkGLError(bool printIfNoError = false);

    static void log(std::string message);
};

#endif //ERROR_UTILS_H
