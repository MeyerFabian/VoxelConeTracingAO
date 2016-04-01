/* All classes that make use of gui should be subclass of this.
Controllable takes pointer to app and registers itself for calling
updateGui() per frame. The fillGui() method must be implemented by
subclasses to fill the gui. */

#ifndef CONTROLLABLE_H_
#define CONTROLLABLE_H_

#include "externals/ImGui/imgui.h"

#include <string>

class App;

class Controllable
{
public:
    Controllable(std::string title);
    Controllable(App* pApp, std::string title);
    virtual ~Controllable() = 0;
    void updateGui();

protected:

    virtual void fillGui() = 0;

private:

    std::string mTitle;
};

#endif // CONTROLLABLE_H_
