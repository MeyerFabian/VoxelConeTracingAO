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
    // TODO
    // Rotate primitive into direction where projection is maximized
    // Move triangle in middle of scene
    // Increase triangle size to fill all pixels at borders
    // Fill output stuff

    gl_Position = vec4(0,0,0,1); // somewhere for rendering
    EmitVertex();

    gl_Position = vec4(0,0,0,1); // somewhere for rendering
    EmitVertex();

    gl_Position = vec4(0,0,0,1); // somewhere for rendering
    EmitVertex();

    EndPrimitive();
}
