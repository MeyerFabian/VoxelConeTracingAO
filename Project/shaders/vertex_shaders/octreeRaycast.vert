#version 430

layout(location = 0) in vec2 vertPos;

out vec3 fragPos;

void main()
{
    // pass-through screen filling quad
    fragPos = vec3(vertPos, 1);
    gl_Position = vec4(vertPos, 0, 1);
}