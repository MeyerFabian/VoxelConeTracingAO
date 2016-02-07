#version 430

layout(location = 0) in float placebo;

out float id;

uniform float volumeExtent;;
layout(r32ui, location = 1) readonly restrict uniform uimageBuffer positionImage;

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
    id = gl_VertexID;
    uint codedPosition = uint(imageLoad(positionImage,int(id)).x);
    vec3 position = volumeExtent * uintXYZ10ToVec3(codedPosition) - volumeExtent/2;
    gl_Position = projection * cameraView * vec4(position, 1);
}
