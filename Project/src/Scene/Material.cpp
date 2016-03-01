#include "Material.h"

#include <fstream>
#include <iostream>
Material::Material(std::string areaName, aiMaterial const * material)
{
    // Name of material
    aiString name;
    material->Get(AI_MATKEY_NAME,name);
    m_name = std::string(name.C_Str());

    // Prepare texture reading
    aiString texturePath;

    // Diffuse
    if (material->GetTexture(aiTextureType_DIFFUSE, 0, &texturePath, NULL, NULL, NULL, NULL, NULL) == AI_SUCCESS)
    {
        m_upDiffuse = std::unique_ptr<Texture>(new Texture(std::string(TEXTURES_PATH) + "/" + areaName + "/" + std::string(texturePath.C_Str())));
    }
}

Material::~Material()
{
    // Nothing to do
}

void Material::bind(ShaderProgram* pShaderProgram) const
{
    // Bind diffuse texture
    if(m_upDiffuse.get() != NULL)
    {
        // Bind texture
        m_upDiffuse->bind(GL_TEXTURE0);

        // Bind slot to correct uniform
        glUniform1i(glGetUniformLocation(pShaderProgram->getShaderProgramHandle(), "tex"), 0);
    }
}
