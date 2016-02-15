
#version 430

/*
* Basic Ambient Occlussion Visualization.
*/


/*
* We will use 5 Cones in this implementation for now.
* Shader Code should be generic for more Cones and actually passing a uniform later.
* But we will stick with this "hardcoded" behavior for now.
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
vec3 cones[NUM_CONES]= vec3[NUM_CONES](
	vec3(0.0,		1.0,		0.0			),
	vec3(0.8660254,	0.5,		0.0			),
	vec3(0.0,		0.5,		0.8660254	),
	vec3(-0.8660254,0.5,		0.0			),
	vec3(0.0,		0.5,		-0.8660254	),
);

//GBuffer-Textures
uniform sampler2D positionTex;
uniform sampler2D normalTex;
uniform sampler2D tangentTex;

//octree-pool
layout(r32ui, location = 0) uniform readonly uimageBuffer octree;
layout(binding = 2) uniform sampler3D brickPool;

//other uniforms
//uniform vec3 eyeVector;
uniform float volumeRes;
uniform vec2 screenSize;
uniform float beginningVoxelSize;
uniform float directionBeginScale;
uniform float volumeExtent;
uniform int maxSteps;

//!< out-
layout(location = 0) out vec4 FragColor;
layout(location = 1) out vec4 Everything_else;

// gl_FragCoord is built in for input Fragment Coordinate (in Pixels),
// divide it by Screensize to get a value between 0..1 to sample our Framebuffer textures 

vec2 calcTexCoord(){
	return gl_FragCoord.xy / screenSize;
}
void calcAmbientOcclusion(){

}

void calcTextureSpace(){
}
void main()
{
    vec2 UVCoord = calcTexCoord();

	vec4 position = texture(positionTex,UVCoord).rgba;
	vec4 normal = texture(normalTex,UVCoord).rgba;
	vec4 tangent = texture(tangentTex,UVCoord).rgba;
	
	Everything_else= tangent * volumeRes *  normal * position * beginningVoxelSize * directionBeginScale *
	volumeExtent * maxSteps;

	float finalColor = (1.0 - (abs(normal.x) +abs(normal.y) +abs(normal.z)) /3.0)*1.25 ;
	FragColor = vec4(finalColor,finalColor,finalColor,1.0); 

}
