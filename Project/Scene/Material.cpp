#include "Material.h"

// TODO: testing
#include <iostream>

Material::Material(aiMaterial const * material)
{
    for(int i = 0; i < material->mNumProperties; i++)
    {
        std::string key = material->mProperties[i]->mKey;

        std::cout << key << std::endl;
    }
}

Material::~Material()
{

}

void Material::bind() const
{

}
