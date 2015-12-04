//
// Created by nils1990 on 04.12.15.
//

#ifndef ERRORUTILS_H
#define ERRORUTILS_H

#include <cuda_runtime.h>
#include <string>
#include "externals/gl3w/include/GL/gl3w.h"
#include "externals/GLFW/include/GLFW/glfw3.h"

#define cudaErrorCheck(ans) { ErrorHandler::gpuAssert((ans), __FILE__, __LINE__); }

    /**
     * @brief The ErrorHandler implements useful methods for cuda error handling
    */
    class ErrorHandler
    {
    public:
        /**
         * @brief throws a CudaException if code != cudaSuccess
         * @details You should use the #define cudaErrorCheck
        */
        static void gpuAssert(cudaError_t code, const char *file, int line);

        static GLenum checkGLError(bool printIfNoError = false);
    };

#endif //ERRORUTILS_H
