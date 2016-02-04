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
out vec4 AABB; // x1, y1, x2, y2

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
    // Calculate normal
    vec3 triNormal =
        abs(
            cross(
                In[1].posDevice - In[0].posDevice,
                In[2].posDevice - In[0].posDevice)); // not normalized

    // Which direction is prominent? (max component of triNormal)
    float triNormalMax = max(max(triNormal.x, triNormal.y), triNormal.z);

    // Now, prominent direction is coded in triNormal
    vec2 pos[3];

    // Ugly case stuff, think about better
    // Rotate in direction of camera
    // Order of vertices not important, no culling used
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
    bool orientationChanged = false;
    vec3 orientationHelper = cross(
                                vec3(pos[1],0) - vec3(pos[0],0),
                                vec3(pos[2],0) - vec3(pos[0],0));
    if(orientationHelper.z < 0)
    {
        // Change orientation of triangle
        vec2 tmp = pos[2];
        pos[2] = pos[1];
        pos[1] = tmp;

        // Don't forget to rescue that information
        orientationChanged = true;
    }

    // Half pixel size
    float halfPixelSize = pixelSize / 2.0;

    // Center of triangle
    vec2 center = (pos[0] + pos[1] + pos[2]) / 3;

    // Prepare variables for line equations
    vec2 lineStarts[3];
    vec2 lineDirections[3];

    // Conservative rasterziation
    for(int i = 0; i <= 2; i++)
    {
        // Go over vertices and create line equation
        lineStarts[i] = pos[i];
        int j = (i+1) % 3;
        lineDirections[i] = normalize(pos[j] - pos[i]);
    }

    // Move lines away from center using cross product
    vec2 expandDirections[3];
    for(int i = 0; i <= 2; i++)
    {
        expandDirections[i] = cross2D(lineDirections[i]); // should be normalized
        lineStarts[i] += expandDirections[i] * halfPixelSize * 1.41f; // TODO: should depend on angle
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
        pos[i] = vec2((c1*b2 - c2*b1)/(a1*b2 -a2*b1), (a1*c2 - a2*c1)/(a1*b2 -a2*b1));
    }

    // Move into center (but only full pixels)
    vec2 centerOffset = -center + vec2(floatMod(center.x, pixelSize), floatMod(center.y, pixelSize));
    pos[0] += centerOffset;
    pos[1] += centerOffset;
    pos[2] += centerOffset;

    // Calculate max/min values for clipping
    float minX = min(pos[2].x, min(pos[0].x, pos[1].x));
    float minY = min(pos[2].y, min(pos[0].y, pos[1].y));
    float maxX = max(pos[2].x, max(pos[0].x, pos[1].x));
    float maxY = max(pos[2].y, max(pos[0].y, pos[1].y));

    // Set bounding box for clipping
    AABB = vec4(
        minX - halfPixelSize,
        minY - halfPixelSize,
        maxX + halfPixelSize,
        maxY + halfPixelSize);

    // Scale bounding box to pixels
    AABB = ((AABB + 1.0) / 2.0) * (2.0 / pixelSize);

    // First vertex
    Out.posDevice = In[0].posDevice;
    Out.normal = In[0].normal;
    Out.uv = In[0].uv;
    gl_Position = vec4(pos[0],0,1);
    EmitVertex();

    // Second vertex
    if(orientationChanged)
    {
        Out.posDevice = In[2].posDevice;
        Out.normal = In[2].normal;
        Out.uv = In[2].uv;
    }
    else
    {
        Out.posDevice = In[1].posDevice;
        Out.normal = In[1].normal;
        Out.uv = In[1].uv;
    }
    gl_Position = vec4(pos[1],0,1);
    EmitVertex();

    // Third vertex
    if(orientationChanged)
    {
        Out.posDevice = In[1].posDevice;
        Out.normal = In[1].normal;
        Out.uv = In[1].uv;
    }
    else
    {
        Out.posDevice = In[2].posDevice;
        Out.normal = In[2].normal;
        Out.uv = In[2].uv;
    }
    gl_Position = vec4(pos[2],0,1);
    EmitVertex();

    EndPrimitive();
}
