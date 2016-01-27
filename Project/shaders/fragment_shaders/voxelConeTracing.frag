
#version 330

/*
* Basic Fragmentshader.
*/

//!< in-variables

//!< uniforms
uniform sampler2D camDepthTex;
uniform sampler2D positionTex;
uniform sampler2D colorTex;
uniform sampler2D normalTex;
uniform sampler2D uvTex;
uniform sampler2D LightViewMapTex;

uniform vec2 screenSize;

//!< out-
layout(location = 0) out vec4 FragColor;
layout(location = 1) out vec4 Everything_else;

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

	float DepthFromLight = texture(LightViewMapTex,UVCoord).r;
	float DepthFromCamera =  texture(camDepthTex,UVCoord).r;
    
	Everything_else= uv*color*normal*position*DepthFromCamera;

    //Show depthmap from the camera
	float VisualDepth = 1.0 - (1.0 - DepthFromLight)*255.0f;
	FragColor = vec4(VisualDepth) ;


}
