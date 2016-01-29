
#version 330

/*
* Basic Fragmentshader.
*/


//!< uniforms
uniform sampler2D camDepthTex;
uniform sampler2D positionTex;
uniform sampler2D colorTex;
uniform sampler2D normalTex;
uniform sampler2D uvTex;
uniform sampler2D LightViewMapTex;

//light uniforms
uniform	vec3 LightPosition;
uniform	vec3 LightColor;
uniform	float LightAmbientIntensity;
uniform	float LightDiffuseIntensity;
uniform float shininess;

uniform mat4 LightModel;
uniform mat4 LightView;
uniform mat4 LightProjection;

//other uniforms
uniform vec3 eyeVector;
uniform vec2 screenSize;

//!< out-
layout(location = 0) out vec4 FragColor;
layout(location = 1) out vec4 Everything_else;

// gl_FragCoord is built in for input Fragment Coordinate (in Pixels),
// divide it by Screensize to get a value between 0..1 to sample our Framebuffer textures 

vec2 calcTexCoord(){
	return gl_FragCoord.xy / screenSize;
}
bool calcOcclusion(vec4 position,vec3 lightdirection,vec3 normal){

	vec4 positionsFromLight = LightProjection * LightView * LightModel * position;
	vec3 ProjCoords = positionsFromLight.xyz / positionsFromLight.w;
	vec2 UVCoords;
	UVCoords.x = 0.5 * ProjCoords.x + 0.5;
	UVCoords.y = 0.5 * ProjCoords.y + 0.5;
	float z = 0.5 *  ProjCoords.z + 0.5;
	if(UVCoords.x <0.0f || UVCoords.y <0.0f || UVCoords.x >1.0f || UVCoords.y >1.0f){
	return false;
	}
	float DepthFromLight = texture(LightViewMapTex,UVCoords).r;
	float bias = 0.00002;  
	if(abs(DepthFromLight - z)< bias){
		return true;
	}
	else{
		return false;
	}
}

vec3 calcLight(vec4 position, vec4 normal){
	vec3 ambientTerm = LightColor * LightAmbientIntensity;
	vec3 diffuseTerm = vec3(0.0f);

	vec3 lightdirection = normalize(LightPosition-position.xyz);
	vec3 finalNormal = normalize(normal.xyz);

	if(calcOcclusion(position,lightdirection,finalNormal)){
		diffuseTerm = LightColor * dot(finalNormal,lightdirection) *LightDiffuseIntensity;
	}

	vec3 lightValue = ambientTerm + diffuseTerm ;
	return lightValue;
}
void main()
{
    vec2 UVCoord = calcTexCoord();
	vec4 color = texture(colorTex,UVCoord).rgba;
	vec4 normal = texture(normalTex,UVCoord).rgba;
	vec4 position = texture(positionTex,UVCoord).rgba;
	vec4 uv = texture(uvTex,UVCoord).rgba;

	float DepthFromCamera =  texture(camDepthTex,UVCoord).r;
    
	Everything_else= uv*color*normal*position*DepthFromCamera
	*vec4(eyeVector,1.0) * LightDiffuseIntensity * vec4(LightPosition,1.0) *shininess;

    //Show depthmap from the camera
	//float VisualDepth = 1.0 - (1.0 - DepthFromCamera)*255.0f;
	//FragColor = vec4(VisualDepth,VisualDepth,VisualDepth,1.0);
	vec3 finalColor = color.xyz * clamp(calcLight(position, normal),0.0,1.0);
	FragColor = vec4(finalColor,1.0); 

}
