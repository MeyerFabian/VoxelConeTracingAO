#version 330

layout(location = 0) in vec3 vertPos;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

out fragPos;

void main()
{
    // passthrough screen filling quad in front of camera
    fragPos = proj * view * model * vertPos;
}