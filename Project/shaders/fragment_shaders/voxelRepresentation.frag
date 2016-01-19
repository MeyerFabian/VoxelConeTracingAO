#version 430

in vec3 fragPos;

uniform vec3 camPos;
//uniform layout(r32ui, location = 1) uimageBuffer positionOutputImage;
uniform float stepSize;

layout(location = 0) out vec4 fragColor;

void main()
{
    vec3 dir = fragPos - camPos;
    normalize(dir);

    fragColor = vec4(fragPos, 1);
}