#version 430

// In / out
layout(location = 0) out vec4 fragColor;
flat in vec4 col;

void main()
{
    fragColor = vec4(col.rgb, 1);
}
