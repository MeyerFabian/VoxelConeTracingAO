#version 430

/*
* Voxelization fragment shader.
*/

//!< in-variables
in RenderVertex
{
    vec3 posDevice;
    vec3 normal;
    vec2 uv;
} In;

//!< uniforms
layout(binding = 0) uniform atomic_uint index;
uniform sampler2D tex;
uniform layout(r32ui, location = 1) uimageBuffer positionOutputImage;
uniform layout(rgba8, location = 2) imageBuffer normalOutputImage;
uniform layout(rgba8, location = 3) imageBuffer colorOutputImage;

//!< out-variables
layout(location = 0) out vec4 fragColor;

void main()
{
    // TODO
    // Clipping (when triangle size was increased)

    // Index in output textures
    uint idx = atomicCounterIncrement(index);

    // Save position of voxel fragment
    // TODO (still from -1 to 1)
    imageStore(positionOutputImage, int(idx), uvec4(1337));

    // Save normal of voxel fragment
    imageStore(normalOutputImage, int(idx), vec4(In.normal, 0));

    // Save color of voxel fragment
    imageStore(colorOutputImage, int(idx), texture(tex, In.uv).rgba);
}
