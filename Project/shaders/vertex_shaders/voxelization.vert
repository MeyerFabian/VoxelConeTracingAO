#version 430

/*
* Voxelization vertex shader.
*/

//!< in-variables
layout(location = 0) in vec4 positionAttribute;
layout(location = 1) in vec2 uvCoordAttribute;
layout(location = 2) in vec4 normalAttribute;

//!< uniforms
uniform mat4 orthographicProjection;
uniform mat4 model;

//!< out-variables
out Vertex
{
    vec3 posDevice;
    vec3 normal;
    vec2 uv;
} Out;

void main()
{
    Out.posDevice = (orthographicProjection * model * positionAttribute).xyz;
    Out.normal = (transpose( inverse( model ) ) *  normalAttribute).xyz;
    Out.uv = uvCoordAttribute;
}
