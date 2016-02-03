#version 430

layout(location = 0) in float placebo;

out float id;

uniform int voxelCount;
uniform layout(r32ui, location = 1) uimageBuffer positionImage;

uniform mat4 projectionView;
uniform mat4 cameraView;

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
        uint codedPosition = uint(imageLoad(positionImage,int(id)).x);
        gl_Position = projectionView * cameraView * vec4(uintXYZ10ToVec3(codedPosition), 1);
    }
    else
    {
        gl_Position = vec4(0, 0, 0, 1);
    }
	
}
