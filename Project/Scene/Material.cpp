#include "Material.h"

// TODO: testing
#include <iostream>

Material::Material(aiMaterial const * material)
{
    for(int i = 0; i < material->mNumProperties; i++)
    {
        std::string key(material->mProperties[i]->mKey.C_Str());

        std::string property(material->mProperties[i]->mData);

        std::cout << key << ": " << property << std::endl;
    }
}

Material::~Material()
{

}

void Material::bind() const
{

}
