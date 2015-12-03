extern "C" // this is not necessary imho, but gives a better idea on where the function comes from
{
    void test();    // test.cuh
}

#include "App.h"
#include "SparseOctree/BrickPool.h"

#include <iostream>
#include <assimp/scene.h>

// has to be included after opengl
#include <cuda.h>
#include <cuda_gl_interop.h>

// Main
int main(void)
{
    std::cout << "hello from gcc" << std::endl;
    test();

    // Init Cuda and enable OpenGL interop
    auto error = cudaGLSetGLDevice(0);

    // Test if a Cuda capable device is available
    if(error != static_cast<cudaError>(CUDA_SUCCESS))
    {
        std::cerr << "no cuda capable device" << std::endl;
    }

    App app;

    BrickPool pool;
    pool.init();

    app.run();
}
