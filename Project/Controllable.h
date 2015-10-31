#ifndef CONTROLLABLE_H_
#define CONTROLLABLE_H_

#include "externals/ImGui/imgui.h"
#include <string>

class App;

class Controllable
{
public:

    Controllable(App* pApp, std::string title);
    virtual ~Controllable() = 0;
    void updateGui();

protected:

    virtual void fillGui() = 0;

private:

    std::string mTitle;
};

#endif // CONTROLLABLE_H_
