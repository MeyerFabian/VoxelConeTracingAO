#include "App.h"
#include "Utilities/errorUtils.h"

// has to be included after opengl
#include <cuda_gl_interop.h>

// Main
int main(void)
{
    // Init Cuda and enable OpenGL interop
    cudaErrorCheck(cudaGLSetGLDevice(0));
    App app;

    app.run();
}
