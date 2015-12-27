#version 430

/*
* Voxelization geometry shader.
*/

//!< in-variables
layout(triangles) in;
layout(triangle_strip, max_vertices = 3) out;

in Vertex
{
    vec3 posWorld;
    vec3 posDevice;
    vec3 normal;
    vec2 uv;
} In[3];

out Voxel
{
    vec3 posWorld;
    vec3 normal;
    vec2 uv;
} Out;

void main()
{
    // ### Calculate direction where projection is maximized ###

    // Calculate normal
    vec3 a = In[1].posDevice - In[0].posDevice;
    vec3 b = In[2].posDevice - In[0].posDevice;
    vec3 triNormal = cross(a,b); // not normalized

    // Which direction is prominent? (max component of triNormal)
    float triNormalMax = max(max(triNormal.x, triNormal.y), triNormal.z);
    triNormal /= triNormalMax;

    // Do some magic, but no if :D

    // TODO
    // Rotate primitive into direction where projection is maximized
    // Move triangle in middle of scene
    // Increase triangle size to fill all pixels at borders
    // Emit vertices and set output values

    gl_Position = vec4(0,0,0,1); // somewhere for rendering
    EmitVertex();

    gl_Position = vec4(0,0,0,1); // somewhere for rendering
    EmitVertex();

    gl_Position = vec4(0,0,0,1); // somewhere for rendering
    EmitVertex();

    EndPrimitive();
}
