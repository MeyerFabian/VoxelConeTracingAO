extern "C" // this is not necessary imho, but gives a better idea on where the function comes from
{
    void test();    // test.cuh
}

#include "App.h"
#include "Utilities/errorUtils.h"

#include <iostream>
#include <assimp/scene.h>

// has to be included after opengl
#include <cuda_gl_interop.h>


// Main
int main(void)
{
    std::cout << "hello from gcc" << std::endl;
    test();

    // Init Cuda and enable OpenGL interop
    cudaErrorCheck(cudaGLSetGLDevice(0));

    App app;

    app.run();
}
