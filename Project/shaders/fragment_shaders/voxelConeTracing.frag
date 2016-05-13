
#version 430

/*
* Basic Fragmentshader.
*/

//!< uniforms
layout(r32ui, location = 0) uniform readonly uimageBuffer octree;
layout(binding = 5) uniform sampler3D brickPool;

//!< uniforms
uniform sampler2D positionTex;
uniform sampler2D colorTex;
uniform sampler2D normalTex;
uniform sampler2D tangentTex;
uniform sampler2D LightViewMapTex;

// Cone Tracing Uniforms
//uniform float beginningVoxelSize;
uniform float directionBeginScale;
uniform float volumeExtent;
uniform int maxSteps;
uniform float maxDistance;
uniform float lambda;
uniform float volumeRes;
uniform float colorBleeding;

//light uniforms
uniform	vec3 LightPosition;
uniform	vec3 LightColor;
uniform	float LightAmbientIntensity;
uniform	float LightDiffuseIntensity;
//uniform float shininess;

uniform mat4 LightModel;
uniform mat4 LightView;
uniform mat4 LightProjection;

//other uniforms
//uniform vec3 eyeVector;
uniform vec2 screenSize;
uniform vec2 shadowToWindowRatio;
uniform float ambientOcclusionScale;
//!< out-
layout(location = 0) out vec4 FragColor;


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

void alphaWeighting(inout float alpha,float distance){
	alpha = 1/(1+lambda*distance)*alpha;
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

	
    vec3 innerOctreePosition = getVolumePos(rayPosition);
	vec3 parentInnerOctreePosition = vec3(0,0,0);
	uint parentNodeOffset = 0;
	uint parentPointer = 0;

    // Reset child pointer
    childPointer = firstChildPointer;

    // Go through octree
    for(int level = 1; level < maxLevel; level++)
    {
        // Determine, in which octant the searched position is
        uvec3 nextOctant = uvec3(0, 0, 0);
        nextOctant.x = uint(2 * innerOctreePosition.x);
        nextOctant.y = uint(2 * innerOctreePosition.y);
        nextOctant.z = uint(2 * innerOctreePosition.z);

		parentNodeOffset = nodeOffset;
		parentPointer = childPointer;
		parentInnerOctreePosition = innerOctreePosition;
		
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
        if(voxelSize >= voxelSizeOnLevel[level+1])
        {
		
			float parentVoxelSize = voxelSizeOnLevel[level];
			float childVoxelSize = voxelSizeOnLevel[level+1];
			// PARENT BRICK SAMPLING
			// Brick coordinates
            uint parentBrickTile = imageLoad(octree, int(parentNodeOffset + parentPointer *16U)+1).x;
            vec3 parentBrickCoords = decodeBrickCoords(parentBrickTile);
 
            // Here we should intersect our brick seperately
            // Go one octant deeper in this inner loop cicle to determine exact brick coordinate
            parentBrickCoords+= 2 * parentInnerOctreePosition;

            // read texture  
			// TODO: im not sure about the volumeRes offset
            vec4 parentSrc = texture(brickPool, parentBrickCoords/volumeRes+ (1.0/volumeRes)/2.0);

			// CHILD BRICK SAMPLING
            // Brick coordinates
            uint brickTile = imageLoad(octree, int(nodeOffset + childPointer *16U)+1).x;
            vec3 brickCoords = decodeBrickCoords(brickTile);

            // Here we should intersect our brick seperately
            // Go one octant deeper in this inner loop cicle to determine exact brick coordinate
            brickCoords += 2 * innerOctreePosition;
            // read texture
            vec4 childSrc = texture(brickPool, brickCoords/volumeRes+ (1.0/volumeRes)/2.0);

			float quadrilinearT = (voxelSize- childVoxelSize)/(parentVoxelSize - childVoxelSize);

            outputColor = (1.0 - quadrilinearT) * childSrc + quadrilinearT * parentSrc;


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
vec4 coneTracing(vec3 perimeterStart,vec3 perimeterDirection,float coneAperture, float cosWeight){
    float distanceTillMainLoop = distanceByVoxelSize(coneAperture,voxelSizeOnLevel[maxLevel]);
	float samplingRate = voxelSizeOnLevel[maxLevel];
	float distance = samplingRate/2.0;
	vec3 rayPosition = vec3(0.0);
	vec4 color = vec4(0.0,0.0,0.0,0.0);
	float voxelSize = voxelSizeOnLevel[maxLevel];
	float alpha = 0.0f;
	float oldSamplingRate =0.0f;
	vec4 premultipliedColor = vec4(0,0,0,0);

	while(distance < distanceTillMainLoop){
		rayPosition = perimeterStart + distance * perimeterDirection;
		vec4 volColor = cosWeight * rayCastOctree(rayPosition,voxelSize);
		alpha = volColor.w;
		distance += samplingRate;
		alphaWeighting(alpha,distance);
		premultipliedColor= vec4(volColor.xyz,1.0) * alpha;
		color =  (1.0 - color.a) * premultipliedColor + color;
	}
	
	while(distance < maxDistance && color.w <0.9){
		voxelSize = voxelSizeByDistance(distance,coneAperture);
		oldSamplingRate = samplingRate;
		samplingRate = voxelSize;
		distance += samplingRate/2.0;
		rayPosition = perimeterStart + distance * perimeterDirection;
		vec4 volColor  = cosWeight * rayCastOctree(rayPosition,voxelSize);
		alpha = volColor.w;
		alphaWeighting(alpha,distance);
		alphaCorrection(alpha,oldSamplingRate,samplingRate);
		distance += samplingRate/2.0;
		premultipliedColor = vec4(volColor.xyz,1.0) * alpha;
		color =  (1.0 - color.a) * premultipliedColor + color;
	}
	
	return color;
}


float calcDirectLightOcclusion(vec4 position,vec3 lightdirection,vec3 normal){

	vec4 positionsFromLight = LightProjection * LightView * LightModel * position;
	vec3 ProjCoords = positionsFromLight.xyz / positionsFromLight.w;
	vec2 UVCoords;
	UVCoords.x = 0.5 * ProjCoords.x + 0.5;
	UVCoords.y = 0.5 * ProjCoords.y + 0.5;
	float z = 0.5 *  ProjCoords.z + 0.5;
	
	float bias = max(0.00001,0.000025 *(1.0f - dot(lightdirection,normal)));
	float DepthFromLight[16];
	float offset= bias * 75.0f;
	float brightness=0.0f;
	
	const int MAX_ITER = 16;

	//Percentage Close Filtering Mask
	vec2 kernel[MAX_ITER]= vec2[MAX_ITER](
	vec2(1.4f* offset,0.0f),
	vec2(1.4f* -offset,0.0f)*1.5f,
	vec2(0.0f,1.4f* offset)*1.5f,
	vec2(0.0f,1.4f* -offset),
	vec2(offset,offset)*1.5f,
	vec2(-offset,offset),
	vec2(offset,-offset),
	vec2(-offset,-offset)*1.5f,
	vec2(2.1f* offset,0.7f*offset)*1.5f,
	vec2(2.1f* offset,0.7f*-offset),
	vec2(0.7f*offset,2.1f*offset),
	vec2(0.7f*-offset,2.1f*-offset)*1.5f,
	vec2(-2.1f* offset,0.7f*offset),
	vec2(-2.1f* offset,0.7f*-offset)*1.5f,
	vec2(-0.7f*offset,2.1f*offset)*1.5f,
	vec2(-0.7f*offset,2.1f*-offset)
	);

	for(int i= 0 ; i< MAX_ITER ;i++){
		vec2 samplePos = UVCoords + kernel[i];
		if(samplePos.x <0.0f || samplePos.y <0.0f || samplePos.x >1.0f || samplePos.y>1.0f){
			break;
		}
		samplePos = samplePos*shadowToWindowRatio;
		DepthFromLight[i]= texture(LightViewMapTex,samplePos).r;
		if(abs(DepthFromLight[i] - z)< bias){
			brightness+= 1.0/MAX_ITER;
		}
		
	}
	
	return brightness;
}

vec4 calcLight(vec4 position, vec4 normal){
	vec3 ambientTerm = clamp(LightColor * LightAmbientIntensity,0.0,1.0);
	vec3 diffuseTerm = vec3(0.0f);

	vec3 lightdirection = normalize(LightPosition-position.xyz);
	vec3 finalNormal = normalize(normal.xyz);

	float brightness = calcDirectLightOcclusion(position,lightdirection,finalNormal);
	diffuseTerm = clamp(brightness* LightColor *  (dot(finalNormal,lightdirection)) *LightDiffuseIntensity,0.0,1.0);
	

	vec3 lightValue = clamp(ambientTerm + diffuseTerm,0.0,1.0) ;
	return vec4(lightValue,1.0);
}

void main()
{
    vec2 UVCoord = calcTexCoord();
	vec4 color = texture(colorTex,UVCoord).rgba;

	// normal and tangent wrap up the tangent space in which we want to transform our cones
	// tangent will be the x-direction, normal is the y-direction
	vec3 tangent = texture(tangentTex,UVCoord).rgb;
	vec4 normal = texture(normalTex,UVCoord).rgba;
	

	// bitangent is calculated by cross product and is our z-direction
	vec3 bitangent = cross(normal.xyz,tangent);

	// we will push them into one matrix to rotate the cone coordinates
	// accordingly to the given normal
	mat3 OutOfTangentSpace = mat3(tangent,normal.xyz,bitangent);

	vec4 position = texture(positionTex,UVCoord).rgba;
    
	vec4 finalColor = color * calcLight(position, normal);

	
	//We precompute voxelsizes on the different levels, 
	for(int level = 0; level <=maxLevel ; level++){
		voxelSizeOnLevel[level] = volumeExtent / (pow2[level])/3.0 *2.0;
	}
	
	//consider loop unrolling
	for(int i = 0 ; i < NUM_CONES;i++){

		//Push the cone a little bit out of the voxel by its normal
		vec3 coneStart = position.xyz + normal.xyz * voxelSizeOnLevel[maxLevel] *directionBeginScale;
		
		
		vec3 coneDirection = OutOfTangentSpace * cones[i];
		float cosWeight = abs(dot(normal.xyz,coneDirection));

		float coneAperture = aperture[i];

		// finalColor will be accumulated due to cone tracing in the octree 
		vec4 tempColor = coneTracing(coneStart, coneDirection,coneAperture,cosWeight) / (NUM_CONES);
		finalColor.xyz -= ambientOcclusionScale* vec3(tempColor.w);
		finalColor.xyz += colorBleeding * tempColor.xyz;
	}
	
	FragColor =  finalColor; 

}
