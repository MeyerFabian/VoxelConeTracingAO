#version 430

uniform int resolution;

// Calculate 3D position from index
vec3 linearTo3D(int id)
{
    int z = id / (resolution * resolution);
    id -= (z * resolution * resolution);
    int y = id / resolution;
    int x = id % resolution;
    return vec3(x,y,z);
}

void main()
{
    // Give geometry shader the voxel coordinate
    gl_Position = vec4(linearTo3D(int(gl_VertexID)), 1);
}
