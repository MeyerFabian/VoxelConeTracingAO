#version 430

/*
* Voxelization geometry shader.
*/

//!< in-variables
layout(triangles) in;
layout(triangle_strip, max_vertices = 3) out;

// Input vertex from vertex shader
in Vertex
{
    vec3 posDevice;
    vec3 normal;
    vec2 uv;
} In[3];

// Vertex ready for rasterizer
out RenderVertex
{
    vec3 posDevice;
    vec3 normal;
    vec2 uv;
} Out;

// Output coordinates of clipping quad (axis aligned bounding box)
flat out vec4 AABB; // x1, y1, x2, y2 in pixel coordinates

//!< uniforms
uniform float pixelSize;

// Cross for 2D
vec2 cross2D(vec2 vector)
{
    return vec2(vector.y, -vector.x);
}

// Float mod
float floatMod(float value, float divisor)
{
    int a = int(value / divisor);
    return (value - (a * divisor));
}

void main()
{
    // Half pixel size
    float halfPixelSize = pixelSize / 2.0;

    // Calculate normal
    vec3 triNormal =
        abs(
            cross(
                In[1].posDevice - In[0].posDevice,
                In[2].posDevice - In[0].posDevice));

    // Which direction is prominent? (max component of triNormal)
    float triNormalMax = max(max(triNormal.x, triNormal.y), triNormal.z);

    vec2 pos[3];
    if(triNormal.x == triNormalMax)
    {
        pos[0] = In[0].posDevice.zy;
        pos[1] = In[1].posDevice.zy;
        pos[2] = In[2].posDevice.zy;
    }
    else if(triNormal.y == triNormalMax)
    {
        pos[0] = In[0].posDevice.xz;
        pos[1] = In[1].posDevice.xz;
        pos[2] = In[2].posDevice.xz;
    }
    else
    {
        pos[0] = In[0].posDevice.xy;
        pos[1] = In[1].posDevice.xy;
        pos[2] = In[2].posDevice.xy;
    }

    // Determine orientation of triangle
    vec3 orientationHelper = cross(
                                vec3(pos[1],0) - vec3(pos[0],0),
                                vec3(pos[2],0) - vec3(pos[0],0));
    if(orientationHelper.z < 0)
    {
        // Change orientation of triangle
        vec2 tmp2;
        vec3 tmp3;

        tmp2 = pos[2];
        pos[2] = pos[1];
        pos[1] = tmp2;

        tmp3 = In[2].posDevice;
        In[2].posDevice = In[1].posDevice;
        In[1].posDevice = tmp3;

        tmp3 = In[2].normal;
        In[2].normal = In[1].normal;
        In[1].normal = tmp3;

        tmp2 = In[2].uv;
        In[2].uv = In[1].uv;
        In[1].uv = tmp2;
    }

    // Set bounding box for clipping (TODO: does not work :-C )
    /*AABB = vec4(
        min(pos[2].x, min(pos[0].x, pos[1].x)) - halfPixelSize,
        min(pos[2].y, min(pos[0].y, pos[1].y)) - halfPixelSize,
        max(pos[2].x, max(pos[0].x, pos[1].x)) + halfPixelSize,
        max(pos[2].y, max(pos[0].y, pos[1].y)) + halfPixelSize); */
    AABB = vec4(-1,-1,1,1); // Just the complete device coordinates...

    // Convert to pixel space
    AABB = (AABB + 1.0) / pixelSize;

    // Prepare variables for line equations
    vec2 lineStarts[3];
    vec2 expandDirections[3];

    // Conservative rasterziation
    for(int i = 0; i <= 2; i++)
    {
        // Move lines away from center using cross product
        int j = (i+1) % 3;
        expandDirections[i] = cross2D(normalize(pos[j] - pos[i]));
        lineStarts[i] = pos[i] + expandDirections[i] * halfPixelSize * 1.41f; // TODO: should depend on angle
    }

    // Cut lines and use found points as output
    for(int i = 0; i <= 2; i++)
    {
        int j = (i+1) % 3;
        float a1 = expandDirections[i].x;
        float b1 = expandDirections[i].y;
        float a2 = expandDirections[j].x;
        float b2 = expandDirections[j].y;
        float c1 = dot(expandDirections[i], lineStarts[i]);
        float c2 = dot(expandDirections[j], lineStarts[j]);
        pos[j] = vec2((c1*b2 - c2*b1)/(a1*b2 -a2*b1), (a1*c2 - a2*c1)/(a1*b2 -a2*b1));
    }

    // First vertex
    Out.posDevice = In[0].posDevice;
    Out.normal = In[0].normal;
    Out.uv = In[0].uv;
    gl_Position = vec4(pos[0],0,1);
    EmitVertex();

    // Second vertex
    Out.posDevice = In[1].posDevice;
    Out.normal = In[1].normal;
    Out.uv = In[1].uv;
    gl_Position = vec4(pos[1],0,1);
    EmitVertex();

    // Third vertex
    Out.posDevice = In[2].posDevice;
    Out.normal = In[2].normal;
    Out.uv = In[2].uv;
    gl_Position = vec4(pos[2],0,1);
    EmitVertex();

    EndPrimitive();
}
