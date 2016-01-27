#version 330
 
 /**
 * This simple shader passes out all important Attributes.
 */

 //!< in-variables
layout(location = 0) in vec4 positionAttribute;


//!< uniforms
uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

//!< out-variables

void main(){

    gl_Position =  projection * view * model * positionAttribute;

}
