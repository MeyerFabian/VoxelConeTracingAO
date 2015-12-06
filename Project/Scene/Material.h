#ifndef MATERIAL_H_
#define MATERIAL_H_

#include "Texture.h"
#include "rendering/ShaderProgram.h"
#include <assimp/scene.h>
#include <memory>

class Material
{
public:

    Material(aiMaterial const * material);
    virtual ~Material();

    void bind(ShaderProgram* pShaderProgram) const;

private:

    // Members
    std::string mName;
    std::unique_ptr<Texture> mupDiffuse;
    std::unique_ptr<ShaderProgram> mupShader;
};

#endif // MATERIAL_H_
