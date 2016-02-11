#version 330
 
 /**
 * This simple shader passes out all important Attributes for rendering into a GBuffer.
 */

 //!< in-variables
layout(location = 0) in vec4 positionAttribute;
layout(location = 1) in vec2 uvCoordAttribute;
layout(location = 2) in vec4 normalAttribute;


//!< uniforms
uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

//!< out-variables
out vec3 passWorldPosition;
out vec2 passUVCoord;
out vec3 passWorldNormal;

void main(){
	
    passUVCoord			= uvCoordAttribute;
    passWorldPosition	= (model * positionAttribute).xyz;
    passWorldNormal		= normalize( ( transpose( inverse( model ) ) * normalAttribute).xyz );
	
	//needed for depth attachment
    gl_Position			= projection * view * model * positionAttribute;

}
