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

//!< out-variables
layout(location = 0) out vec4 fragColor;

void main()
{
    vec4 color = texture(tex, In.uv).rgba;

    // TODO
    // Clipping (when triangle size was increased)
    // Do something with atomic counter
    // Write something to buffer texture
    // Color, position...

    atomicCounterIncrement(index);

    color.rgb += In.normal; // Just to have it used
    fragColor = color; // Get rid of that
}
