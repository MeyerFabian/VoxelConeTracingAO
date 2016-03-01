#ifndef MATERIAL_H_
#define MATERIAL_H_

#include "Texture.h"
#include <Rendering/ShaderProgram.h>
#include <assimp/scene.h>

#include <memory>

class Material
{
public:

    Material(std::string areaName, aiMaterial const * material);
    virtual ~Material();

    void bind(ShaderProgram* pShaderProgram) const;

private:

    // Members
    std::string m_name;
    std::unique_ptr<Texture> m_upDiffuse;
    std::unique_ptr<ShaderProgram> m_upShader;
};

#endif // MATERIAL_H_
