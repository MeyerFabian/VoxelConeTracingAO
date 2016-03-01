#version 430

layout(r32ui, location = 1) readonly restrict uniform uimageBuffer positionImage;
layout(rgba8, location = 2) readonly restrict uniform imageBuffer normalImage;
layout(rgba8, location = 3) readonly restrict uniform imageBuffer colorImage;

uniform mat4 cameraView;
uniform mat4 projection;
uniform float volumeExtent;

out vec3 col;

vec3 uintXYZ10ToVec3(uint val)
{
    float x = float(uint(val & uint(0x000003FF))) / 1023.0;
    float y = float(uint((val >> 10U)& uint(0x000003FF))) / 1023.0;
    float z = float(uint((val >> 20U)& uint(0x000003FF))) / 1023.0;
    return vec3(x,y,z);
}


void main()
{
    uint codedPosition = uint(imageLoad(positionImage,int(gl_VertexID)).x);
    vec3 position = volumeExtent * uintXYZ10ToVec3(codedPosition) - volumeExtent/2;
    gl_Position = projection * cameraView * vec4(position, 1);

    // If normal is not read, image location is not found etc. Much error
    col = imageLoad(colorImage,int(gl_VertexID)).rgb + 0.01 * imageLoad(normalImage,int(gl_VertexID)).rgb;
}
