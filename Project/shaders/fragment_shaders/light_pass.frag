
#version 330

/*
* Basic Fragmentshader.
*/

//!< in-variables

//!< uniforms
uniform sampler2D positionTex;
uniform sampler2D colorTex;
uniform sampler2D normalTex;
uniform sampler2D uvTex;
uniform sampler2D CamDepthTex;

uniform vec2 screenSize;

//!< out-
layout(location = 0) out vec4 fragColor;


// gl_FragCoord is built in for input Fragment Coordinate (in Pixels),
// divide it by Screensize to get a value between 0..1 to sample our Framebuffer textures 

vec2 calcTexCoord(){
	return gl_FragCoord.xy / screenSize;
}

void main()
{
    vec2 UVCoord = calcTexCoord();
	vec4 color = texture(colorTex,UVCoord).rgba;
	vec4 normal = texture(normalTex,UVCoord).rgba;
	vec4 position = texture(positionTex,UVCoord).rgba;
	vec4 uv = texture(uvTex,UVCoord).rgba;
	
	float DepthFromCamera = 1.0 - (1.0 - texture(CamDepthTex,UVCoord).x);
    

    fragColor = vec4(DepthFromCamera);
}
