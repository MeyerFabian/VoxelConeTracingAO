#ifndef MATERIAL_H_
#define MATERIAL_H_

#include <assimp/scene.h>

class Material
{
public:

    Material(aiMaterial const * material);
    virtual ~Material();

    void bind() const;

private:


};

#endif // MATERIAL_H_
