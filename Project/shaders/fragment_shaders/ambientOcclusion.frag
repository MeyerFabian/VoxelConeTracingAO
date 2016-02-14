
#version 430

/*
* Basic Fragmentshader.
*/


//!< uniforms
layout(r32ui, location = 0) uniform readonly uimageBuffer octree;
layout(binding = 3) uniform sampler3D brickPool;

uniform sampler2D positionTex;
uniform sampler2D normalTex;
uniform sampler2D tangentTex;


//other uniforms
//uniform vec3 eyeVector;
uniform float volumeRes;
uniform vec2 screenSize;
uniform float beginningVoxelSize;
uniform float directionBeginScale;
uniform float volumeExtent;
uniform float maxDistance;

//!< out-
layout(location = 0) out vec4 FragColor;
layout(location = 1) out vec4 Everything_else;

// gl_FragCoord is built in for input Fragment Coordinate (in Pixels),
// divide it by Screensize to get a value between 0..1 to sample our Framebuffer textures 
vec2 calcTexCoord(){
	return gl_FragCoord.xy / screenSize;
}

/*
*			OCTREE DEFINES AND FUNCTIONS
*/

const int maxLevel = 8;
const uint pow2[] = {1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024};
// Helper
uint getBit(uint value, uint position)
{
    return (value >> (position-1)) & 1u;
}

uvec3 decodeBrickCoords(uint coded)
{
    uvec3 coords;
    coords.z =  coded & 0x000003FF;
    coords.y = (coded & 0x000FFC00) >> 10U;
    coords.x = (coded & 0x3FF00000) >> 20U;
    return coords;
}

vec3 getVolumePos(vec3 worldPos)
{
    return (worldPos / volumeExtent) + 0.5;
}



/*
*			CONE TRACING FUNCTIONS AND DEFINES
*/

const int NUM_CONES = 5;

vec3 cones[NUM_CONES]	=vec3[NUM_CONES](
	vec3(0.0,		1.0,	0.0			),
	vec3(0.866025,	0.5,	0			),
	vec3(0.0,		0.5,	0.866025	),
	vec3(-0.866025,	0.5,	0			),
	vec3(0.0,		0.5,	-0.866025	)
);
float aperture[NUM_CONES]={
	60.0,
	60.0,
	60.0,
	60.0,
	60.0};

/*	
*	@param	distance                    Distance from the apex
*	@param	coneAperture                Aperture of the given cone in degree(angle)
*	@return voxelsize                   corresponding to the distance
*	Calculates the voxel size we will be looking up 
*	by the distance of that voxel from the apex
*/
float voxelSizeByDistance(float distance, float coneAperture){
	float halfAperture = coneAperture /2.0;
        float voxelSize = tan(halfAperture) * distance * 2.0;
	return voxelSize;
}

/*
*	@param	coneAperture                Aperture of the given cone in Degree(angle)
*	@param	voxelsize                   corresponding to the distance
*	@return distance                    corresponding to the voxelsize
*	Calculates the initial distance for a given voxel size
*	Used initially to find the first voxel on the maximum resolution
*	that can be looked up in the octree.
*/
float distanceByVoxelSize(float coneAperture, float voxelSize){
	float halfAperture = coneAperture /2.0;
    float distance = voxelSize / 2.0 / tan(radians(halfAperture));
	return distance;
}

/*
*	@param alpha                        alpha-value of the accumulated color before correcting
*	@param oldSamplingDistance          Sampling Distance we sampled with before
*	@param newSamplingDistance          Sampling Distance we use in this step
*	Corrects the alpha-value when using adaptive Sampling
*/
void alphaCorrection(	inout float alpha, 
						float oldSamplingDistance, 
						float newSamplingDistance){

						alpha = 1-		pow( 1-alpha, 
										newSamplingDistance/oldSamplingDistance 
										);
}

vec4 rayCastOctree(vec3 rayPosition){
	vec4 outputColor = vec4(0,0,0,0);

    // Octree reading preparation
    uint nodeOffset = 0;
    uint childPointer = 0;
    uint nodeTile;

    // Get first child pointer
    nodeTile = imageLoad(octree, int(0)).x;
    uint firstChildPointer = nodeTile & uint(0x3fffffff);

	
    vec3 innerOctreePosition = getVolumePos(rayPosition);
    // Reset child pointer
    childPointer = firstChildPointer;

    // Go through octree
    for(int j = 1; j < maxLevel; j++)
    {
        // Determine, in which octant the searched position is
        uvec3 nextOctant = uvec3(0, 0, 0);
        nextOctant.x = uint(2 * innerOctreePosition.x);
        nextOctant.y = uint(2 * innerOctreePosition.y);
        nextOctant.z = uint(2 * innerOctreePosition.z);

        // Make the octant position 1D for the linear memory
        nodeOffset = 2 * (nextOctant.x + 2 * nextOctant.y + 4 * nextOctant.z);
        nodeTile = imageLoad(octree, int(childPointer * 16U + nodeOffset)).x;

        // Update position in volume
        innerOctreePosition.x = 2 * innerOctreePosition.x - nextOctant.x;
        innerOctreePosition.y = 2 * innerOctreePosition.y - nextOctant.y;
        innerOctreePosition.z = 2 * innerOctreePosition.z - nextOctant.z;

        // The 32nd bit indicates whether the node has children:
        // 1 means has children
        // 0 means does not have children
        // Only read from brick, if we are at aimed level in octree
        if(getBit(nodeTile, 32) == 0)
        {

            // Brick coordinates
            uint brickTile = imageLoad(octree, int(nodeOffset + childPointer *16U)+1).x;
            vec3 brickCoords = decodeBrickCoords(brickTile);

            
            // Here we should intersect our brick seperately
            // Go one octant deeper in this inner loop cicle to determine exact brick coordinate
            brickCoords.x += 2 * innerOctreePosition.x;
            brickCoords.y += 2 * innerOctreePosition.y;
            brickCoords.z += 2 * innerOctreePosition.z;


            // Accumulate color
            vec4 src = texture(brickPool, brickCoords/volumeRes+ (1.0/volumeRes)/2.0);

            outputColor = src;


            // Break inner loop
            break;
        }
        else
        {
            // If the node has children we read the pointer to the next nodetile
            childPointer = nodeTile & uint(0x3fffffff);
        }
    }

	return outputColor;
}

// perimeterDirection seems to be calulcated right :)
vec4 coneTracing(vec4 perimeterStart,vec3 perimeterDirection,float coneAperture){
    float voxelSizeOnLowestLevel = volumeExtent / (pow2[maxLevel]);
    float distanceTillMainLoop = distanceByVoxelSize(coneAperture,voxelSizeOnLowestLevel);
	float samplingRate= voxelSizeOnLowestLevel;
	float distance = samplingRate/2.0;
	vec3 rayPosition = vec3(0.0);
	vec4 color = vec4(0.0,0.0,0.0,0.0);
	while(distance < distanceTillMainLoop){
		rayPosition = perimeterStart.xyz + 0.0f * distance * perimeterDirection;
		vec4 interpolatedColor = rayCastOctree(rayPosition);
		distance += samplingRate;
		color += interpolatedColor;
	}
	/*
	while(distance < maxDistance){
		rayPosition = perimeterStart.xyz + distance * perimeterDirection;
		vec4 interpolatedColor = rayCastOctree(rayPosition);
		distance += samplingRate;
		//voxelSize = voxelSizeByDistance(distance,coneAperture);
		color += interpolatedColor;
	}
	*/
	return color;
}

/*
*						MAIN
*/
void main()
{
    vec2 UVCoord = calcTexCoord();

	vec4 position = texture(positionTex,UVCoord).rgba;

	// normal and tangent wrap up the tangent space in which we want to transform our cones
	// tangent will be the x-direction, normal is the y-direction
	vec3 tangent = texture(tangentTex,UVCoord).rgb;
	vec3 normal = texture(normalTex,UVCoord).rgb;
	
	// bitangent is calculated by cross product and is our z-direction
	vec3 bitangent = cross(normal,tangent);

	// we will push them into one matrix to rotate the cone coordinates
	// accordingly to the given normal
	mat3 OutOfTangentSpace = mat3(tangent,normal,bitangent);

	vec4 finalColor = vec4(0.0,0.0,0.0,0.0);
	
	//consider loop unrolling
	for(int i = 0 ; i < NUM_CONES;i++){
		vec3 coneDirection = OutOfTangentSpace * cones[i];
		float coneAperture = aperture[i];
		// finalColor will be accumulated due to cone tracing in the octree 
		finalColor += coneTracing(position, coneDirection,coneAperture) / (NUM_CONES);
	}

	Everything_else=vec4(tangent,1.0) * vec4(normal,1.0)*volumeRes *  position*beginningVoxelSize*directionBeginScale*
	volumeExtent*maxDistance;

	FragColor = vec4(finalColor); 

}
