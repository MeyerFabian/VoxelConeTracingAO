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
uniform mat4 modelNormal;
uniform mat4 projectionView;

//!< out-variables
out Vertex
{
    vec3 posWorld;
    vec3 posDevice;
    vec3 normal;
    vec2 uv;
} Out;

void main()
{
    Out.posWorld = (model * positionAttribute).xyz;
    Out.posDevice = (projectionView * vec4(Out.posWorld, 1)).xyz;
    Out.normal = (modelNormal * normalAttribute).xyz;
    Out.uv = uvCoordAttribute;
}
