#version 430

/*
* Voxelization geometry shader.
*/

//!< in-variables
layout(triangles) in;
layout(triangle_strip, max_vertices = 3) out;

in Vertex
{
    vec3 posDevice;
    vec3 normal;
    vec2 uv;
} In[3];

out RenderVertex
{
    vec3 posDevice;
    vec3 normal;
    vec2 uv;
} Out;

void main()
{
    // TODO
    // Increase triangle size to fill all pixels at borders

    // ### Calculate direction where projection is maximized ###

    // Calculate normal
    vec3 triNormal =
        abs(
            cross(
                In[1].posDevice - In[0].posDevice,
                In[2].posDevice - In[0].posDevice)); // not normalized

    // Which direction is prominent? (max component of triNormal)
    float triNormalMax = max(max(triNormal.x, triNormal.y), triNormal.z);
    triNormal /= triNormalMax;
    triNormal = floor(triNormal);

    // Now, prominent direction is coded in triNormal
    vec2 a;
    vec2 b;
    vec2 c;

    // Ugly case stuff, think about better
    // Rotate in direction of camera
    if(triNormal.x > 0)
    {
        a = In[0].posDevice.zy;
        b = In[1].posDevice.zy;
        c = In[2].posDevice.zy;
    }
    else if(triNormal.y > 0)
    {
        a = In[0].posDevice.xz;
        b = In[1].posDevice.xz;
        c = In[2].posDevice.xz;
    }
    else
    {
        a = In[0].posDevice.xy;
        b = In[1].posDevice.xy;
        c = In[2].posDevice.xy;
    }

    // Move into center
    vec2 center = (a + b + c) / 3;
    a -= center;
    b -= center;
    c -= center;

    // First vertex
    Out.posDevice = In[0].posDevice;
    Out.normal = In[0].normal;
    Out.uv = In[0].uv;
    gl_Position = vec4(a,0,1);
    EmitVertex();

    // Second vertex
    Out.posDevice = In[1].posDevice;
    Out.normal = In[1].normal;
    Out.uv = In[1].uv;
    gl_Position = vec4(b,0,1);
    EmitVertex();

    // Third vertex
    Out.posDevice = In[2].posDevice;
    Out.normal = In[2].normal;
    Out.uv = In[2].uv;
    gl_Position = vec4(c,0,1);
    EmitVertex();

    EndPrimitive();
}
