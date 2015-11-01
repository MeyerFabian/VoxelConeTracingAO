/* At the moment only test to show how to make subclass of Controllable. */

#ifndef VOXELIZATION_H_
#define VOXELIZATION_H_

#include "Controllable.h"

class Voxelization : public Controllable
{
public:

    Voxelization(App* pApp) : Controllable(pApp, "Voxelization") {}

protected:

    virtual void fillGui(); // Implementation of Controllable

private:


};

#endif // VOXELIZATION_H_
