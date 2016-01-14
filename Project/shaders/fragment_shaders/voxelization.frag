#version 430

/*
* Voxelization fragment shader.
*/

// kopier aus dem github von den unileuten
uint vec3ToUintXYZ10(uvec3 val)
{
    return (uint(val.z) & 0x000003FF)   << 20U
            |(uint(val.y) & 0x000003FF) << 10U
            |(uint(val.x) & 0x000003FF);
}

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

    // Position from 0 to 1023 in volume
    uvec3 pos = uvec3(((In.posDevice + 1) / 2.0) * 1024);
    //uint codedPos = (pos.x << 20) | (pos.y << 10) | (pos.z);

    // Save position of voxel fragment
    imageStore(positionOutputImage, int(idx), uvec4(vec3ToUintXYZ10(pos)));

    // Save normal of voxel fragment
    imageStore(normalOutputImage, int(idx), vec4(In.normal, 0));

    // Save color of voxel fragment
    imageStore(colorOutputImage, int(idx), texture(tex, In.uv).rgba);
}
