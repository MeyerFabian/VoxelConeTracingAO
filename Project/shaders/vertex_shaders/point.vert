#version 430

out vec3 col;

uniform float volumeExtent;
layout(binding = 1, r32ui) readonly restrict uniform uimageBuffer positionImage;
layout(binding = 2, r32ui) readonly restrict uniform uimage3D colorVolume;
layout(binding = 3, r32ui) readonly restrict uniform uimage3D normalVolume;

uniform mat4 projection;
uniform mat4 cameraView;

vec3 uintXYZ10ToVec3(uint val)
{
    float x = float(uint(val & uint(0x000003FF))) / 1023.0;
    float y = float(uint((val >> 10U)& uint(0x000003FF))) / 1023.0;
    float z = float(uint((val >> 20U)& uint(0x000003FF))) / 1023.0;
    return vec3(x,y,z);
}

vec4 RGBA8ToVec4(uint val) {
    return vec4( float((val & 0x000000FF)),
                 float((val & 0x0000FF00) >> 8U),
                 float((val & 0x00FF0000) >> 16U),
                 float((val & 0xFF000000) >> 24U));
}

void main()
{
    // Extract position
    uint codedPosition = uint(imageLoad(positionImage,int(gl_VertexID)).x);
    vec3 relativePosition = uintXYZ10ToVec3(codedPosition);
    vec3 position = volumeExtent * relativePosition - volumeExtent/2;
    gl_Position = projection * cameraView * vec4(position, 1);

    // Extract color
    float res = int(imageSize(colorVolume).x); // Normal volume should have same resolution
    col = (RGBA8ToVec4(uint(imageLoad(colorVolume, ivec3(relativePosition * res))))).xyz;
    col /= 255; // Convert from stored range to display range
}