#include "Voxelization.h"

void Voxelization::fillGui()
{
    // TODO
    int i = 4;
    ImGui::SliderInt("resolution", &i, 1, 10);
}
