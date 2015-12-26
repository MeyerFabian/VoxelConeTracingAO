#version 430

/*
* Voxelization vertex shader.
*/

//!< in-variables
layout(location = 0) in vec4 positionAttribute;
layout(location = 1) in vec2 uvCoordAttribute;
layout(location = 2) in vec4 normalAttribute;

//!< uniforms
uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

//!< out-variables
out Vertex
{
    vec3 posWorld;
    vec3 normal;
    vec2 uv;
} Out;

out vec3 passWorldPosition;
out vec3 passPosition;
out vec2 passUVCoord;

void main()
{
    // TODO
    // Some output pos in device coordinates for later rendering

    Out.posWorld = (model * positionAttribute).xyz;
    Out.normal = normalAttribute.xyz; // TODO matrix multiplication
    Out.uv = uvCoordAttribute;
}
