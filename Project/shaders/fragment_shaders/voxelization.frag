
#version 430

/*
* Voxelization fragment shader.
*/

//!< in-variables
in RenderVertex
{
    vec3 posDevice;
    vec3 normal;
    vec2 uv;
} In;

flat in vec4 AABB;

//!< uniforms
layout(binding = 0) uniform atomic_uint index;
uniform sampler2D tex;
layout(binding = 1, r32ui) restrict writeonly uniform uimageBuffer positionOutputImage;
layout(binding = 2, r32ui) restrict volatile uniform uimage3D colorVolume;
layout(binding = 3, r32ui) restrict volatile uniform uimage3D normalVolume;

//!< out-variables
layout(location = 0) out vec4 fragColor;

//!< const-variables
const uint MAX_NUM_ACC_ITERATIONS = 100;

// Convert vec3 to uint with 10 bit per component (wants values from 0..1023)
uint vec3ToUintXYZ10(uvec3 val)
{
    return (uint(val.z) & 0x000003FF)   << 20U
            |(uint(val.y) & 0x000003FF) << 10U
            |(uint(val.x) & 0x000003FF);
}

// Convert 32bit uint RGBA to vec4
vec4 RGBA8ToVec4(uint val) {
    return vec4( float((val & 0x000000FF)),
                 float((val & 0x0000FF00) >> 8U),
                 float((val & 0x00FF0000) >> 16U),
                 float((val & 0xFF000000) >> 24U));
}

// Convert vec4 to 32bit uint RGBA
uint Vec4ToRGBA8(vec4 val) {
    // Alpha, blue, green, red...
    return (uint(val.w) & 0x000000FF)   << 24U
            |(uint(val.z) & 0x000000FF) << 16U
            |(uint(val.y) & 0x000000FF) << 8U
            |(uint(val.x) & 0x000000FF);
}

// Accumluation for color (Same as for normal but with other image. Faster than extra parameter)
void accumulateColor(vec3 value, ivec3 coords)
{
    // Scale value to 0..255 (in that range it is saved to the integer value)
    vec4 newValue = vec4(value * 255,1); // One is the weight of this value

    // Converted value, ready to be filled into image
    uint newValueU = Vec4ToRGBA8(newValue);

    // Some variables need if there is already something at that voxel
    uint lastValueU = 0;
    uint currentValueU;
    vec4 currentValue;
    uint numIterations = 0;

    // Tries to overwrite image value at each iteration. When successful, loop is left before entering
    while(
        (currentValueU = imageAtomicCompSwap(colorVolume, coords, lastValueU, newValueU)) != lastValueU
        && numIterations < MAX_NUM_ACC_ITERATIONS) {

        // Ok, this time we had no luck. So use the current value from image to create accumulated value for next try
        lastValueU = currentValueU;
        currentValue = RGBA8ToVec4(currentValueU);

        // Denormalize current value from atomic swap call
        currentValue.xyz *= currentValue.a;

        // Add our own value
        currentValue += newValue;

        // Renormalize to fit into 8 bits per component
        currentValue.xyz /= currentValue.a;

        // Convert back to uint for next try
        newValueU = Vec4ToRGBA8(currentValue);

        // Count tries
        numIterations++;
    }
}

// Accumluation for normal (Same as for color but with other image. Faster than extra parameter)
void accumulateNormal(vec3 value, ivec3 coords)
{
    // Scale value to 0..255 (in that range it is saved to the integer value)
    vec4 newValue = vec4(value * 255,1); // One is the weight of this value

    // Converted value, ready to be filled into image
    uint newValueU = Vec4ToRGBA8(newValue);

    // Some variables need if there is already something at that voxel
    uint lastValueU = 0;
    uint currentValueU;
    vec4 currentValue;
    uint numIterations = 0;

    // Tries to overwrite image value at each iteration. When successful, loop is left before entering
    while(
        (currentValueU = imageAtomicCompSwap(normalVolume, coords, lastValueU, newValueU)) != lastValueU
        && numIterations < MAX_NUM_ACC_ITERATIONS) {

        // Ok, this time we had no luck. So use the current value from image to create accumulated value for next try
        lastValueU = currentValueU;
        currentValue = RGBA8ToVec4(currentValueU);

        // Denormalize current value from atomic swap call
        currentValue.xyz *= currentValue.a;

        // Add our own value
        currentValue += newValue;

        // Renormalize to fit into 8 bits per component
        currentValue.xyz /= currentValue.a;

        // Convert back to uint for next try
        newValueU = Vec4ToRGBA8(currentValue);

        // Count tries
        numIterations++;
    }
}

void main()
{
    /* NOT USED
    // Clipping with bounding box
    if( gl_FragCoord.x < AABB.x
        || gl_FragCoord.x >= AABB.z
        || gl_FragCoord.y < AABB.y
        || gl_FragCoord.y >= AABB.w)
    {
        discard;
    }
    */

    // Index in output textures
    uint idx = atomicCounterIncrement(index);

    // Store position
    In.posDevice.z = -In.posDevice.z; // Left hand to right hand system
    vec3 relativePosition = (In.posDevice + 1) * 0.5;
    uvec3 position10Bit = uvec3(relativePosition* 1023);
    imageStore(positionOutputImage, int(idx), uvec4(vec3ToUintXYZ10(position10Bit)));

    // Coordinate in color and normal volume
    float res = int(imageSize(colorVolume).x); // Normal volume should have same resolution
    ivec3 coords = ivec3((vec3(position10Bit) / 1023.0) * res);

    // Accumulate color
    accumulateColor(texture(tex, In.uv).rgb, coords);

    // Accumulate normal
    accumulateNormal(In.normal, coords);
}