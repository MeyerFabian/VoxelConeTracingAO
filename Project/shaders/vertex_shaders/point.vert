#version 430

layout(location = 0) in float placebo;

out int id;

uniform int voxelCount;
uniform layout(r32ui, location = 1) uimageBuffer positionImage;

vec3 uintXYZ10ToVec3(uint val)
{
    float x = float(uint(val & 0x000003FF)) / 1023.0;
    float y = float(uint((val >> 10U)& 0x000003FF)) / 1023.0;
    float z = float(uint((val >> 20U)& 0x000003FF)) / 1023.0;
    return vec3(x,y,z);
}


void main()
{
    // Get id from OpenGL
    id = gl_VertexID;

    if(voxelCount < gl_VertexID)
    {
        // Get position of point out of image
        uint codedPosition = uint(imageLoad(positionImage, id).x);
        gl_Position = vec4(uintXYZ10ToVec3(codedPosition), 1);
    }
    else
    {
        gl_Position = vec4(0, 0, 0, 1);
    }
}
