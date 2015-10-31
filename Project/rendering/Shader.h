#ifndef SHADER_H
#define SHADER_H

#include "externals/gl3w/include/GL/gl3w.h"
#include "externals/GLFW/include/GLFW/glfw3.h"
#include <string>

class Shader
{
public:
    Shader(const GLuint &type);
    ~Shader();

    void loadFromString(const std::string &sourceString);
    void loadFromFile(const std::string &filename);

    void compile();

    inline GLuint getId()           {return m_id;}
    inline std::string getSource()  {return m_source;}

private:
    GLuint m_id;
    std::string m_typeString;
    std::string m_source;
};

#endif //SHADER_H