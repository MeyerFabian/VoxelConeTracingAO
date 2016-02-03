#version 430

in float id;
layout(location = 0) out vec4 fragColor;

uniform float voxelCount;
uniform layout(rgba8, location = 2) imageBuffer normalImage;
uniform layout(rgba8, location = 3) imageBuffer colorImage;

void main()
{
    vec3 normal = imageLoad(normalImage,int(id)).rgb;
    vec3 color = imageLoad(colorImage,int(id)).rgb;
    fragColor = vec4(color + 0.01 * normal, 1); // TODO: make controllable
}
