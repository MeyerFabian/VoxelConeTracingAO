#version 330

in vec3 fragPos;

uniform in vec3 camPos;
uniform in vec3 view;
uniform in int levelOfDetail;
uniform layout(r32ui, location = 1) uimageBuffer positionOutputImage;

void main()
{
    vec3 rayDirection = fragPos - camPos;
    for(int i = 0; i < levelOfDetail; i++)
    {
        fragPos += rayDirection;

    }
}