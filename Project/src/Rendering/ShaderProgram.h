#ifndef SHADER_PROGRAM_H
#define SHADER_PROGRAM_H

#include <vector>
#include <map>

#include "Shader.h"
#include "externals/GLM/glm/glm.hpp"


class ShaderProgram
{

public:
	ShaderProgram(std::string vertexshader, std::string fragmentshader);
	ShaderProgram(std::string vertexshader, std::string fragmentshader, std::string geometryshader);
	~ShaderProgram();

	GLint getShaderProgramHandle();

	ShaderProgram* updateUniform(std::string name, bool value);
	ShaderProgram* updateUniform(std::string name, int value);
	ShaderProgram* updateUniform(std::string name, float value);
	ShaderProgram* updateUniform(std::string name, double value);
	ShaderProgram* updateUniform(std::string name, glm::ivec2 vector);
	ShaderProgram* updateUniform(std::string name, glm::ivec3 vector);
	ShaderProgram* updateUniform(std::string name, glm::ivec4 vector);
	ShaderProgram* updateUniform(std::string name, glm::vec2 vector);
	ShaderProgram* updateUniform(std::string name, glm::vec3 vector);
	ShaderProgram* updateUniform(std::string name, glm::vec4 vector);
	ShaderProgram* updateUniform(std::string name, glm::mat2 matrix);
	ShaderProgram* updateUniform(std::string name, glm::mat3 matrix);
	ShaderProgram* updateUniform(std::string name, glm::mat4 matrix);
	ShaderProgram* updateUniform(std::string name, std::vector<glm::vec2> vector);
	ShaderProgram* updateUniform(std::string name, std::vector<glm::vec3> vector);
	ShaderProgram* updateUniform(std::string name, std::vector<glm::vec4> vector);


	int addBuffer(const std::string &bufferName);
	int addTexture(const std::string &textureName, const std::string &path);
	void addTexture(const std::string &textureName, GLuint textureHandle);
	int addUniform(const std::string &uniformName);

	virtual void use();
	void disable();

	inline std::map<std::string,int>* getUniformMap()	{return &m_uniformMap;}
	inline std::map<std::string,int>* getBufferMap()	{return &m_bufferMap;}
	inline std::map<std::string,int>* getTextureMap()	{return &m_textureMap;}
	
private:
	void readUniforms();
	void readOutputs(Shader& fragmentShader);
	void attachShader(Shader shader);
	void link();

	GLuint uniform(const std::string &uniform);
	GLuint buffer(const std::string &buffer);
	GLuint texture(const std::string &texture);

	GLuint m_shaderProgramHandle;
	int m_shaderCount;

	std::map<std::string,int> m_uniformMap;
	std::map<std::string,int> m_bufferMap;
	std::map<std::string,int> m_textureMap;
};

#endif // SHADER_PROGRAM_H