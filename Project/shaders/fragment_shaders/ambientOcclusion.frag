
#version 430

/*
* Basic Fragmentshader.
*/


uniform sampler2D positionTex;
uniform sampler2D normalTex;
uniform sampler2D tangentTex;
//!< uniforms
layout(r32ui, location = 0) uniform readonly uimageBuffer octree;
layout(binding = 2) uniform sampler3D brickPool;


//other uniforms
//uniform vec3 eyeVector;
uniform float volumeRes;
uniform vec2 screenSize;
uniform float beginningVoxelSize;
uniform float directionBeginScale;
uniform float volumeExtent;
uniform int maxDistance;

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

// perimeterDirection seems to be calulcated right :)
vec4 coneTracing(vec4 perimeterStart,vec3 perimeterDirection,float coneAperture){
    float voxelSizeOnLowestLevel = volumeExtent / (pow2[maxLevel]);
    float distanceTillMainLoop = distanceByVoxelSize(coneAperture,voxelSizeOnLowestLevel);
	float samplingRate= voxelSizeOnLowestLevel;
	float distance = samplingRate/2.0;
	vec3 rayPosition = vec3(0.0);
	vec4 color = vec4(0.0,0.0,0.0,1.0);
	while(distance < distanceTillMainLoop){
		rayPosition = perimeterStart.xyz + distance * perimeterDirection;
		//vec4 interpolatedColor = rayCastOctree(rayPosition);
		distance += samplingRate;
		color += vec4(perimeterDirection,0.0);
	}
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

	vec4 finalColor = vec4(0.0);
	
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
