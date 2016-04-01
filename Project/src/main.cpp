/* Main method of application. Initializes cuda. Creates app and runs it. */

#include "App.h"
#include "Utilities/errorUtils.h"

// Has to be included after opengl
#include <cuda_gl_interop.h>

// Main
int main(void)
{
    // Init Cuda and enable OpenGL interop
    cudaErrorCheck(cudaGLSetGLDevice(0));

    // Run app
    App app;
    app.run();
}
