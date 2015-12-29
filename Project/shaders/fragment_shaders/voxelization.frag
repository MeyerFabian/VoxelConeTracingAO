#version 430

/*
* Voxelization fragment shader.
*/

//!< in-variables
in Voxel
{
    vec3 posWorld;
    vec3 normal;
    vec2 uv;
} In;

//!< uniforms
layout(binding = 0) uniform atomic_uint index;
uniform sampler2D tex;
uniform layout(rgba8, location = 1) imageBuffer colorOutputImage; // location = 1 is ok? no plan

//!< out-variables
layout(location = 0) out vec4 fragColor;

void main()
{
    // TODO
    // Clipping (when triangle size was increased)

    // Index in output textures
    uint idx = atomicCounterIncrement(index);

    // Save color of voxel fragment
   // imageStore(colorOutputImage, int(idx), texture(tex, In.uv).rgba);
   imageStore(colorOutputImage, int(idx), vec4(0,1,0.5,1));
}
