
#version 330
 
 /**
 * This simple shader does a passthrough of the screen filling quad.
 */

 //!< in-variables
layout(location = 0) in vec4 positionAttribute;

//!< uniforms
uniform mat4 identity;

void main(){
    gl_Position =  identity *  positionAttribute;
}