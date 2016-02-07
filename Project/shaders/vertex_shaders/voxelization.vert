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

//!< out-variables
out Vertex
{
    vec3 posDevice;
    vec3 normal;
    vec2 uv;
} Out;

void main()
{
    Out.posDevice = (orthographicProjection * positionAttribute).xyz;
    Out.normal = normalAttribute.xyz;
    Out.uv = uvCoordAttribute;
}
