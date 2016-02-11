
#version 430

/*
* Basic Fragmentshader.
*/


uniform sampler2D positionTex;
uniform sampler2D normalTex;
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
void main()
{
    vec2 UVCoord = calcTexCoord();

	vec4 position = texture(positionTex,UVCoord).rgba;
	vec4 normal = texture(normalTex,UVCoord).rgba;
	
	Everything_else=volumeRes *  normal* position*beginningVoxelSize*directionBeginScale*
	volumeExtent*maxSteps;

	float finalColor = (1.0 - (abs(normal.x) +abs(normal.y) +abs(normal.z)) /3.0)*1.25 ;
	FragColor = vec4(finalColor,finalColor,finalColor,1.0); 

}
