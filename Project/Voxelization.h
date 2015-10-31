#ifndef VOXELIZATION_H_
#define VOXELIZATION_H_

#include "Controllable.h"

// Example how to use Controllable

class Voxelization : public Controllable
{
public:

    Voxelization(App* pApp) : Controllable(pApp, "Voxelization") {}

protected:

    virtual void fillGui();

private:


};

#endif // VOXELIZATION_H_
