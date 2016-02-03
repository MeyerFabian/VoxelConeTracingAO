#version 430

in float id;
layout(location = 0) out vec4 fragColor;

uniform int voxelCount;
//uniform layout(rgba8, location = 2) imageBuffer normalOutputImage;
//uniform layout(rgba8, location = 3) imageBuffer colorOutputImage;

void main()
{
    fragColor = vec4(1,float(id)/voxelCount,0,1);
}
