#version 430

/*
* Voxelization geometry shader.
*/

//!< in-variables
layout(triangles) in;
layout(triangle_strip, max_vertices = 3) out;

struct TempVertex
{
    vec3 posDevice;
    vec3 normal;
    vec2 uv;
};

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

	TempVertex temp[3];
	for(int i=0;i<3;i++){
	temp[i] = TempVertex(In[i].posDevice,In[i].normal,In[i].uv);
	}
    if(orientationHelper.z < 0)
    {
        // Change orientation of triangle
        vec2 tmp2;
        vec3 tmp3;

        tmp2 = pos[2];
        pos[2] = pos[1];
        pos[1] = tmp2;

        tmp3 = In[2].posDevice;
        temp[2].posDevice = temp[1].posDevice;
        temp[1].posDevice = tmp3;

        tmp3 = In[2].normal;
        temp[2].normal = temp[1].normal;
        temp[1].normal = tmp3;

        tmp2 = In[2].uv;
        temp[2].uv = temp[1].uv;
        temp[1].uv = tmp2;
    }

    // Set bounding box for clipping
    AABB = vec4(
        min(pos[2].x, min(pos[0].x, pos[1].x)) - halfPixelSize,
        min(pos[2].y, min(pos[0].y, pos[1].y)) - halfPixelSize,
        max(pos[2].x, max(pos[0].x, pos[1].x)) + halfPixelSize,
        max(pos[2].y, max(pos[0].y, pos[1].y)) + halfPixelSize);

    // Convert to pixel space
    AABB = (AABB + 1.0) / pixelSize;

    // Compute edges of triangle
    /*vec2 diag = vec2(halfPixelSize, halfPixelSize);
    vec2 e0 = pos[1].xy - pos[0].xy;
    vec2 e1 = pos[2].xy - pos[1].xy;
    vec2 e2 = pos[0].xy - pos[2].xy;
    vec2 n0 = normalize(cross2D(e0));
    vec2 n1 = normalize(cross2D(e1));
    vec2 n2 = normalize(cross2D(e2));
    vec2 n0_abs = abs(n0);
    vec2 n1_abs = abs(n1);
    vec2 n2_abs = abs(n2);
    float d20 = dot(n2, e0);
    float d01 = dot(n0, e1);
    float d12 = dot(n1, e2);
    float d02 = dot(n0, e2);
    float d10 = dot(n1, e0);
    float d21 = dot(n2, e1);

    // Expand triangle
    pos[0].xy += e0 * dot(n2_abs, diag) / d20 + e2 * dot(n0_abs, diag) / d02;
    pos[1].xy += e1 * dot(n0_abs, diag) / d01 + e0 * dot(n1_abs, diag) / d10;
    pos[2].xy += e2 * dot(n1_abs, diag) / d12 + e1 * dot(n2_abs, diag) / d21; */

    // First vertex
    Out.posDevice = temp[0].posDevice;
    Out.normal = temp[0].normal;
    Out.uv = temp[0].uv;
    gl_Position = vec4(pos[0],0,1);
    EmitVertex();

    // Second vertex
    Out.posDevice = temp[1].posDevice;
    Out.normal = temp[1].normal;
    Out.uv = temp[1].uv;
    gl_Position = vec4(pos[1],0,1);
    EmitVertex();

    // Third vertex
    Out.posDevice = temp[2].posDevice;
    Out.normal = temp[2].normal;
    Out.uv = temp[2].uv;
    gl_Position = vec4(pos[2],0,1);
    EmitVertex();

    EndPrimitive();
}

// #### Muellers conservative rasterization ###

// Compute edges of triangle
/*vec2 diag = vec2(halfPixelSize, halfPixelSize);
vec2 e0 = pos[1].xy - pos[0].xy;
vec2 e1 = pos[2].xy - pos[1].xy;
vec2 e2 = pos[0].xy - pos[2].xy;
vec2 n0 = normalize(cross2D(e0));
vec2 n1 = normalize(cross2D(e1));
vec2 n2 = normalize(cross2D(e2));
vec2 n0_abs = abs(n0);
vec2 n1_abs = abs(n1);
vec2 n2_abs = abs(n2);
float d20 = dot(n2, e0);
float d01 = dot(n0, e1);
float d12 = dot(n1, e2);
float d02 = dot(n0, e2);
float d10 = dot(n1, e0);
float d21 = dot(n2, e1);

// Expand triangle
pos[0].xy += e0 * dot(n2_abs, diag) / d20 + e2 * dot(n0_abs, diag) / d02;
pos[1].xy += e1 * dot(n0_abs, diag) / d01 + e0 * dot(n1_abs, diag) / d10;
pos[2].xy += e2 * dot(n1_abs, diag) / d12 + e1 * dot(n2_abs, diag) / d21;*/

// ### Raphaels conservative rasterization ###

// Prepare variables for line equations
//vec2 lineStarts[3];
//vec2 expandDirections[3];

// Conservative rasterziation
/*for(int i = 0; i <= 2; i++)
{
    // Move lines away from center using cross product
    int j = (i+1) % 3;
    expandDirections[i] = cross2D(normalize(pos[j] - pos[i]));
    lineStarts[i] = pos[i] + expandDirections[i] * halfPixelSize * 1.41f; // TODO: should depend on angle
}*/

// ### UNROLLED FOR LOOP ABOVE ###
/*expandDirections[0] = cross2D(normalize(pos[1] - pos[0]));
lineStarts[0] = pos[0] + expandDirections[0] * halfPixelSize * 1.41f; // TODO: should depend on angle
expandDirections[1] = cross2D(normalize(pos[2] - pos[1]));
lineStarts[1] = pos[1] + expandDirections[1] * halfPixelSize * 1.41f; // TODO: should depend on angle
expandDirections[2] = cross2D(normalize(pos[0] - pos[2]));
lineStarts[2] = pos[2] + expandDirections[2] * halfPixelSize * 1.41f; // TODO: should depend on angle*/

// Cut lines and use found points as output
/*for(int i = 0; i <= 2; i++)
{
    int j = (i+1) % 3;
    float a1 = expandDirections[i].x;
    float b1 = expandDirections[i].y;
    float a2 = expandDirections[j].x;
    float b2 = expandDirections[j].y;
    float c1 = dot(expandDirections[i], lineStarts[i]);
    float c2 = dot(expandDirections[j], lineStarts[j]);
    pos[j] = vec2((c1*b2 - c2*b1)/(a1*b2 -a2*b1), (a1*c2 - a2*c1)/(a1*b2 -a2*b1));
}*/

// ### UNROLLED FOR LOOP ABOVE ###
/*float a1, b1, a2, b2, c1, c2;

a1 = expandDirections[0].x;
b1 = expandDirections[0].y;
a2 = expandDirections[1].x;
b2 = expandDirections[1].y;
c1 = dot(expandDirections[0], lineStarts[0]);
c2 = dot(expandDirections[1], lineStarts[1]);
pos[1] = vec2((c1*b2 - c2*b1)/(a1*b2 -a2*b1), (a1*c2 - a2*c1)/(a1*b2 -a2*b1));

a1 = expandDirections[1].x;
b1 = expandDirections[1].y;
a2 = expandDirections[2].x;
b2 = expandDirections[2].y;
c1 = c2;
c2 = dot(expandDirections[2], lineStarts[2]);
pos[2] = vec2((c1*b2 - c2*b1)/(a1*b2 -a2*b1), (a1*c2 - a2*c1)/(a1*b2 -a2*b1));

a1 = expandDirections[2].x;
b1 = expandDirections[2].y;
a2 = expandDirections[0].x;
b2 = expandDirections[0].y;
c1 = c2;
c2 = dot(expandDirections[0], lineStarts[0]);
pos[0] = vec2((c1*b2 - c2*b1)/(a1*b2 -a2*b1), (a1*c2 - a2*c1)/(a1*b2 -a2*b1)); */
