#version 430

in float id;
layout(location = 0) out vec4 fragColor;

uniform float voxelCount;
uniform layout(rgba8, location = 2) imageBuffer normalImage;
uniform layout(rgba8, location = 3) imageBuffer colorImage;

void main()
{
    fragColor = vec4(imageLoad(colorImage,int(id)).rgb, 1);
}
