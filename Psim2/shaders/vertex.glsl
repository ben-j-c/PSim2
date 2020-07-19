attribute vec3 vPos;
attribute vec3 vColour;

varying vec4 fColour;

uniform mat4 viewMatrix;
uniform mat4 projectionMatrix;

void main(){
	gl_Position = projectionMatrix*viewMatrix*vec4(vPos, 1.0);
	fColour = vec4(vColour, 0.5);
}