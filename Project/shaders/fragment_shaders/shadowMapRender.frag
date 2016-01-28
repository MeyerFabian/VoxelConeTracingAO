
#version 330

/*
* Basic Fragmentshader.
*/

//!< in-variables

//!< uniforms
uniform sampler2D LightViewMapTex;

uniform vec2 screenSize;

//!< out-
layout(location = 0) out vec4 FragColor;

// gl_FragCoord is built in for input Fragment Coordinate (in Pixels),
// divide it by Screensize to get a value between 0..1 to sample our Framebuffer textures 

vec2 calcTexCoord(){
	return gl_FragCoord.xy / screenSize;
}

void main()
{
    vec2 UVCoord = calcTexCoord();
    
	float DepthFromLight = texture(LightViewMapTex,UVCoord).r;
	float VisualDepth = 1.0 - (1.0 - DepthFromLight)*255.0f;
	FragColor = vec4(VisualDepth,VisualDepth,VisualDepth, 1.0) ;


}
