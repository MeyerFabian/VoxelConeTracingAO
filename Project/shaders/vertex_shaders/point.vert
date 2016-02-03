#version 430

layout(location = 0) in float placebo;

out float id;

uniform vec3 volumeCenter;
uniform float volumeExtent;
uniform float voxelCount;
uniform layout(r32ui, location = 1) uimageBuffer positionImage;

uniform mat4 projection;
uniform mat4 cameraView;

vec3 uintXYZ10ToVec3(uint val)
{
    float x = float(uint(val & uint(0x000003FF))) / 1023.0;
    float y = float(uint((val >> 10U)& uint(0x000003FF))) / 1023.0;
    float z = float(uint((val >> 20U)& uint(0x000003FF))) / 1023.0;
    return vec3(x,y,z);
}


void main()
{
    if(gl_VertexID < voxelCount)
    {
        id = gl_VertexID;

        // Get position of point out of image
        uint codedPosition = uint(imageLoad(positionImage,int(id)).x);
        vec3 position = volumeExtent * uintXYZ10ToVec3(codedPosition) - volumeExtent/2;
        position += volumeCenter;
        gl_Position = projection * cameraView * vec4(position, 1);
    }
    else
    {
        id = 0;
        gl_Position = vec4(0, 0, 0, 1);
    }

}
