#version 430

uniform float volumeExtent;

uniform int resolution;

// Calculate 3D position from index
vec3 linearTo3D(int id)
{
    int z = id / (resolution * resolution);
    id -= (z * resolution * resolution);
    int y = id / resolution;
    int x = id % resolution;
    return vec3(x,y,z) / resolution;
}

void main()
{
    vec3 position = volumeExtent * linearTo3D(int(gl_VertexID)) - volumeExtent/2;
    gl_Position = vec4(position, 1);
}
