
#version 430

/*
* Basic Ambient Occlussion Visualization.
*/

/*
* We will use 5 Cones in this implementation for now.
* Shader Code should be generic for more Cones and actually passing a uniform later.
* But we will stick with this "hardcoded" behavior for now.
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

/*
* Cones have an aperture of 60 degrees.
* First Cone uses the normal as direction.
* The other 4 Cones are placed onto the x-z plane.
* So on the x and y axis we use full 90 degrees.
* (60 from one of the 4 and 30 from the top one)
* We could actually fit two more Cones into the spaces.
* Direction for the cones can for instance be calculated 
* by unit sphere parametrization (swapping y and z):
* v1(0°,0°),v2(0°,60°),v3(90°,60°)
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

	
float voxelSizeOnLevel[maxLevel+1];

/*	
*	@param	distance                    Distance from the apex
*	@param	coneAperture                Aperture of the given cone in degree(angle)
*	@return voxelsize                   corresponding to the distance
*	Calculates the voxel size we will be looking up 
*	by the distance of that voxel from the apex
*/
float voxelSizeByDistance(float distance, float coneAperture){
	float halfAperture = coneAperture /2.0;
    float voxelSize = tan(radians(halfAperture)) * distance * 2.0;
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

vec4 rayCastOctree(vec3 rayPosition,float voxelSize){
	vec4 outputColor = vec4(0,0,0,0);

    // Octree reading preparation
    uint nodeOffset = 0;
    uint childPointer = 0;
    uint nodeTile;

    // Get first child pointer
    nodeTile = imageLoad(octree, int(0)).x;
    uint firstChildPointer = nodeTile & uint(0x3fffffff);

	vec3 parentInnerOctreePosition = vec3(0,0,0);
	uint parentNodeOffset = 0;
	uint parentPointer = 0;
	
    vec3 innerOctreePosition = getVolumePos(rayPosition);

    // Reset child pointer
    childPointer = firstChildPointer;

    // Go through octree
    for(int level = 1; level < maxLevel; level++)
    {
        // Determine, in which octant the searched position is
        vec3 nextOctant = vec3(0, 0, 0);
        nextOctant.x = floor(2 * innerOctreePosition.x);
        nextOctant.y = floor(2 * innerOctreePosition.y);
        nextOctant.z = floor(2 * innerOctreePosition.z);

		//Update ParentInfo => Make previous child parent now
		parentNodeOffset = nodeOffset;
		parentPointer = childPointer;
		parentInnerOctreePosition = innerOctreePosition;

		

        // Make the octant position 1D for the linear memory
        nodeOffset = uint(2 * (nextOctant.x + 2 * nextOctant.y + 4 * nextOctant.z));
        nodeTile = imageLoad(octree, int(childPointer * 16U + nodeOffset)).x;

        // Update position in volume
        innerOctreePosition.x = 2 * innerOctreePosition.x - nextOctant.x;
        innerOctreePosition.y = 2 * innerOctreePosition.y - nextOctant.y;
        innerOctreePosition.z = 2 * innerOctreePosition.z - nextOctant.z;
		
		if(voxelSize >= voxelSizeOnLevel[level+1])
		{
		
			float parentVoxelSize = voxelSizeOnLevel[level];
			float childVoxelSize = voxelSizeOnLevel[level+1];


			// PARENT BRICK SAMPLING
			// Brick coordinates
			uint parentBrickTile = imageLoad(octree, int(parentNodeOffset + parentPointer *16U)+1).x;
			vec3 parentBrickCoords = decodeBrickCoords(parentBrickTile);

			// CHILD BRICK SAMPLING
			// Brick coordinates
			uint brickTile = imageLoad(octree, int(nodeOffset + childPointer *16U)+1).x;
			vec3 brickCoords = decodeBrickCoords(brickTile);

			if(getBit(brickTile, 31) == 1)
            {
			// Here we should intersect our brick seperately
			// Go one octant deeper in this inner loop cicle to determine exact brick coordinate
			parentBrickCoords +=  2.0*parentInnerOctreePosition;

			// Here we should intersect our brick seperately
			// Go one octant deeper in this inner loop cicle to determine exact brick coordinate
			brickCoords += 2.0* innerOctreePosition;

			vec4 parentSrc = texture(brickPool, parentBrickCoords/volumeRes + (1.0/volumeRes)/2.0);

			vec4 childSrc = texture(brickPool, brickCoords/volumeRes + (1.0/volumeRes)/2.0);
			
			float quadrilinearT = (voxelSize- childVoxelSize)/(parentVoxelSize - childVoxelSize);

			outputColor = (1.0 - quadrilinearT) * childSrc + quadrilinearT * parentSrc;
			}
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
vec4 coneTracing(vec3 perimeterStart,vec3 perimeterDirection,float coneAperture,float samplingDistanceModifier){
    float distanceTillMainLoop = distanceByVoxelSize(coneAperture,voxelSizeOnLevel[maxLevel]);
	float samplingRate = voxelSizeOnLevel[maxLevel];
	float distance = samplingRate/2.0*samplingDistanceModifier;
	vec3 rayPosition = vec3(0.0);
	vec4 color = vec4(0.0,0.0,0.0,0.0);
	float voxelSize = voxelSizeOnLevel[maxLevel];

	while(distance < distanceTillMainLoop){
		rayPosition = perimeterStart + distance * perimeterDirection;
		vec4 interpolatedColor = rayCastOctree(rayPosition,voxelSize);
		distance += samplingRate;
		color += interpolatedColor;
	}
	while(distance < maxDistance){
		voxelSize = voxelSizeByDistance(distance,coneAperture);
		samplingRate = voxelSize;
		distance += samplingRate/2.0*samplingDistanceModifier;
		rayPosition = perimeterStart + distance * perimeterDirection;
		vec4 interpolatedColor = rayCastOctree(rayPosition,voxelSize);
		distance += samplingRate/2.0*samplingDistanceModifier;
		color += interpolatedColor;
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

	vec4 finalColor = vec4(0.0,0.0,0.0,0.0);

	//We precompute voxelsizes on the different levels, 
	for(int level = 0; level<=maxLevel ; level++){
		voxelSizeOnLevel[level] = volumeExtent / (pow2[level]);
	}

	//consider loop unrolling
	for(int i = 0 ; i < NUM_CONES;i++){

		//Push the cone a little bit out of the voxel by its normal
		vec3 coneStart = position.xyz + normal * voxelSizeOnLevel[maxLevel] *directionBeginScale;
		
		vec3 coneDirection = OutOfTangentSpace * cones[i];

		/*
		* Target: scale the samplingDistance of a cone by its relative angle to the voxel grid axes
		* Why? Reduces sampling artifats because the voxel grid is orthogonal but our sampling is not.
		* We set the coneDirection into the first octant and calculate the distance between the x-axis.
		* Will be somewhere inbetween 0 and 90 degrees.
		* We actually only want to restrict ourselves to angles of < 45 degrees, which splits the octant in half again.
		* The inverted cos of the angle between the adjusted coneDirection and x-Axis is the scale we want to adjust our samplingDistance to.
		*/
		float angleAxisCone = acos(dot(vec3(abs(coneDirection.x),abs(coneDirection.y),abs(coneDirection.z)), vec3(1,0,0))); 
		if(angleAxisCone >= 45.0){
			angleAxisCone = 90.0 - angleAxisCone;
		}
		float samplingDistanceModifier = 1.0;

		if(angleAxisCone >=1.0){
		samplingDistanceModifier = 1.0/(cos(angleAxisCone));
		}

		float coneAperture = aperture[i];

		// finalColor will be accumulated due to cone tracing in the octree 
		finalColor += coneTracing(coneStart, coneDirection,coneAperture,samplingDistanceModifier) / (NUM_CONES);
	}

	Everything_else=vec4(tangent,1.0) * vec4(normal,1.0)*volumeRes *  position*beginningVoxelSize*directionBeginScale*
	volumeExtent*maxDistance;

	FragColor = vec4(finalColor); 

}
