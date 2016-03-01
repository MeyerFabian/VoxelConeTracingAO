#version 430

layout(location = 0) in vec2 vertPos;

void main()
{
    // Pass-through screen filling quad
    gl_Position = vec4(vertPos, 0, 1);
}
