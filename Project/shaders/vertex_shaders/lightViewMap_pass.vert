#version 330
 
 /**
 * Shader passes out Depth Map of the Lights View.
 */

 //!< in-variables
layout(location = 0) in vec4 positionAttribute;


//!< uniforms
uniform mat4 LightView;
uniform mat4 LightProjection;

void main(){

    gl_Position =  LightProjection * LightView * positionAttribute;

}
