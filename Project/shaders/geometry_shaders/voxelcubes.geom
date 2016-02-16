#version 430

layout(points) in;
layout(triangle_strip, max_vertices = 3) out;

uniform mat4 projection;
uniform mat4 cameraView;

void main()
{
    // Size
    vec2 halfSize = vec2(1, 1);

    // Matrix
    mat4 matrix = projection * cameraView;

    // Emit quad
    gl_Position = matrix * (gl_in[0].gl_Position + vec4(-halfSize.x, -halfSize.y, 0, 0));
    EmitVertex();

    gl_Position = matrix * (gl_in[0].gl_Position + vec4(halfSize.x, -halfSize.y, 0, 0));
    EmitVertex();

    gl_Position = matrix * (gl_in[0].gl_Position + vec4(0, halfSize.y, 0, 0));
    EmitVertex();

    EndPrimitive();

}
