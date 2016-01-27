#ifndef OCTREE_MIPMAPPING_CUH
#define OCTREE_MIPMAPPING_CUH

__device__ void filterBrick(const uint3 &brickCoords)
{
    // TODO: filter brick
    uint3 insertPositions[8];
    // front corners
    insertPositions[0] = make_uint3(0,0,0);
    insertPositions[1] = make_uint3(2,0,0);
    insertPositions[2] = make_uint3(2,2,0);
    insertPositions[3] = make_uint3(0,2,0);

    //back corners
    insertPositions[4] = make_uint3(0,0,2);
    insertPositions[5] = make_uint3(2,0,2);
    insertPositions[6] = make_uint3(2,2,2);
    insertPositions[7] = make_uint3(0,2,2);

    uchar4 colors[8];
    colors[0] = make_uchar4(0,0,0,0);
    colors[1] = make_uchar4(0,0,0,0);
    colors[2] = make_uchar4(0,0,0,0);
    colors[3] = make_uchar4(0,0,0,0);
    colors[4] = make_uchar4(0,0,0,0);
    colors[5] = make_uchar4(0,0,0,0);
    colors[6] = make_uchar4(0,0,0,0);
    colors[7] = make_uchar4(0,0,0,0);

    surf3Dread(&colors[0], surfRef, (insertPositions[0].x + brickCoords.x) * sizeof(uchar4), insertPositions[0].y + brickCoords.y, insertPositions[0].z + brickCoords.z);
    surf3Dread(&colors[1], surfRef, (insertPositions[1].x + brickCoords.x) * sizeof(uchar4), insertPositions[1].y + brickCoords.y, insertPositions[1].z + brickCoords.z);
    surf3Dread(&colors[2], surfRef, (insertPositions[2].x + brickCoords.x) * sizeof(uchar4), insertPositions[2].y + brickCoords.y, insertPositions[2].z + brickCoords.z);
    surf3Dread(&colors[3], surfRef, (insertPositions[3].x + brickCoords.x) * sizeof(uchar4), insertPositions[3].y + brickCoords.y, insertPositions[3].z + brickCoords.z);
    surf3Dread(&colors[4], surfRef, (insertPositions[4].x + brickCoords.x) * sizeof(uchar4), insertPositions[4].y + brickCoords.y, insertPositions[4].z + brickCoords.z);
    surf3Dread(&colors[5], surfRef, (insertPositions[5].x + brickCoords.x) * sizeof(uchar4), insertPositions[5].y + brickCoords.y, insertPositions[5].z + brickCoords.z);
    surf3Dread(&colors[6], surfRef, (insertPositions[6].x + brickCoords.x) * sizeof(uchar4), insertPositions[6].y + brickCoords.y, insertPositions[6].z + brickCoords.z);
    surf3Dread(&colors[7], surfRef, (insertPositions[7].x + brickCoords.x) * sizeof(uchar4), insertPositions[7].y + brickCoords.y, insertPositions[7].z + brickCoords.z);

    // ################ center: #######################
    float4 tmp = make_float4(0,0,0,0);

    tmp.x += static_cast<unsigned int>(colors[0].x);
    tmp.y += static_cast<unsigned int>(colors[0].y);
    tmp.z += static_cast<unsigned int>(colors[0].z);
    tmp.w += static_cast<unsigned int>(colors[0].w);

    tmp.x += static_cast<unsigned int>(colors[1].x);
    tmp.y += static_cast<unsigned int>(colors[1].y);
    tmp.z += static_cast<unsigned int>(colors[1].z);
    tmp.w += static_cast<unsigned int>(colors[1].w);

    tmp.x += static_cast<unsigned int>(colors[2].x);
    tmp.y += static_cast<unsigned int>(colors[2].y);
    tmp.z += static_cast<unsigned int>(colors[2].z);
    tmp.w += static_cast<unsigned int>(colors[2].w);

    tmp.x += static_cast<unsigned int>(colors[3].x);
    tmp.y += static_cast<unsigned int>(colors[3].y);
    tmp.z += static_cast<unsigned int>(colors[3].z);
    tmp.w += static_cast<unsigned int>(colors[3].w);

    tmp.x += static_cast<unsigned int>(colors[4].x);
    tmp.y += static_cast<unsigned int>(colors[4].y);
    tmp.z += static_cast<unsigned int>(colors[4].z);
    tmp.w += static_cast<unsigned int>(colors[4].w);

    tmp.x += static_cast<unsigned int>(colors[5].x);
    tmp.y += static_cast<unsigned int>(colors[5].y);
    tmp.z += static_cast<unsigned int>(colors[5].z);
    tmp.w += static_cast<unsigned int>(colors[5].w);

    tmp.x += static_cast<unsigned int>(colors[6].x);
    tmp.y += static_cast<unsigned int>(colors[6].y);
    tmp.z += static_cast<unsigned int>(colors[6].z);
    tmp.w += static_cast<unsigned int>(colors[6].w);

    tmp.x += static_cast<unsigned int>(colors[7].x);
    tmp.y += static_cast<unsigned int>(colors[7].y);
    tmp.z += static_cast<unsigned int>(colors[7].z);
    tmp.w += static_cast<unsigned int>(colors[7].w);

    tmp.x *= 0.125f;
    tmp.y *= 0.125f;
    tmp.z *= 0.125f;
    tmp.w *= 0.125f;

    uint3 newCoords = make_uint3(1,1,1);
    newCoords.x+=brickCoords.x;
    newCoords.y+=brickCoords.x;
    newCoords.z+=brickCoords.x;

    surf3Dwrite(make_uchar4(tmp.x,tmp.y,tmp.z,tmp.w), surfRef, newCoords.x*sizeof(uchar4), newCoords.y, newCoords.z);

    // ################### FACES ##########################
    // right side: 1, 2, 5, 6
    tmp = make_float4(0,0,0,0);
    tmp.x += static_cast<unsigned int>(colors[1].x);
    tmp.y += static_cast<unsigned int>(colors[1].y);
    tmp.z += static_cast<unsigned int>(colors[1].z);
    tmp.w += static_cast<unsigned int>(colors[1].w);

    tmp.x += static_cast<unsigned int>(colors[2].x);
    tmp.y += static_cast<unsigned int>(colors[2].y);
    tmp.z += static_cast<unsigned int>(colors[2].z);
    tmp.w += static_cast<unsigned int>(colors[2].w);

    tmp.x += static_cast<unsigned int>(colors[5].x);
    tmp.y += static_cast<unsigned int>(colors[5].y);
    tmp.z += static_cast<unsigned int>(colors[5].z);
    tmp.w += static_cast<unsigned int>(colors[5].w);

    tmp.x += static_cast<unsigned int>(colors[6].x);
    tmp.y += static_cast<unsigned int>(colors[6].y);
    tmp.z += static_cast<unsigned int>(colors[6].z);
    tmp.w += static_cast<unsigned int>(colors[6].w);

    tmp.x *= 0.25f;
    tmp.y *= 0.25f;
    tmp.z *= 0.25f;
    tmp.w *= 0.25f;

    newCoords = make_uint3(2,1,1);
    newCoords.x+=brickCoords.x;
    newCoords.y+=brickCoords.x;
    newCoords.z+=brickCoords.x;

    surf3Dwrite(make_uchar4(tmp.x,tmp.y,tmp.z,tmp.w), surfRef, newCoords.x*sizeof(uchar4), newCoords.y, newCoords.z);

    // left side: 0, 3, 4, 7
    tmp = make_float4(0,0,0,0);
    tmp.x += static_cast<unsigned int>(colors[0].x);
    tmp.y += static_cast<unsigned int>(colors[0].y);
    tmp.z += static_cast<unsigned int>(colors[0].z);
    tmp.w += static_cast<unsigned int>(colors[0].w);

    tmp.x += static_cast<unsigned int>(colors[3].x);
    tmp.y += static_cast<unsigned int>(colors[3].y);
    tmp.z += static_cast<unsigned int>(colors[3].z);
    tmp.w += static_cast<unsigned int>(colors[3].w);

    tmp.x += static_cast<unsigned int>(colors[4].x);
    tmp.y += static_cast<unsigned int>(colors[4].y);
    tmp.z += static_cast<unsigned int>(colors[4].z);
    tmp.w += static_cast<unsigned int>(colors[4].w);

    tmp.x += static_cast<unsigned int>(colors[7].x);
    tmp.y += static_cast<unsigned int>(colors[7].y);
    tmp.z += static_cast<unsigned int>(colors[7].z);
    tmp.w += static_cast<unsigned int>(colors[7].w);

    tmp.x *= 0.25f;
    tmp.y *= 0.25f;
    tmp.z *= 0.25f;
    tmp.w *= 0.25f;

    newCoords = make_uint3(0,1,1);
    newCoords.x+=brickCoords.x;
    newCoords.y+=brickCoords.x;
    newCoords.z+=brickCoords.x;

    surf3Dwrite(make_uchar4(tmp.x,tmp.y,tmp.z,tmp.w), surfRef, newCoords.x*sizeof(uchar4), newCoords.y, newCoords.z);

    // bottom side: 2, 3, 6, 7
    tmp = make_float4(0,0,0,0);
    tmp.x += static_cast<unsigned int>(colors[2].x);
    tmp.y += static_cast<unsigned int>(colors[2].y);
    tmp.z += static_cast<unsigned int>(colors[2].z);
    tmp.w += static_cast<unsigned int>(colors[2].w);

    tmp.x += static_cast<unsigned int>(colors[3].x);
    tmp.y += static_cast<unsigned int>(colors[3].y);
    tmp.z += static_cast<unsigned int>(colors[3].z);
    tmp.w += static_cast<unsigned int>(colors[3].w);

    tmp.x += static_cast<unsigned int>(colors[6].x);
    tmp.y += static_cast<unsigned int>(colors[6].y);
    tmp.z += static_cast<unsigned int>(colors[6].z);
    tmp.w += static_cast<unsigned int>(colors[6].w);

    tmp.x += static_cast<unsigned int>(colors[7].x);
    tmp.y += static_cast<unsigned int>(colors[7].y);
    tmp.z += static_cast<unsigned int>(colors[7].z);
    tmp.w += static_cast<unsigned int>(colors[7].w);

    tmp.x *= 0.25f;
    tmp.y *= 0.25f;
    tmp.z *= 0.25f;
    tmp.w *= 0.25f;

    newCoords = make_uint3(1,2,1);
    newCoords.x+=brickCoords.x;
    newCoords.y+=brickCoords.x;
    newCoords.z+=brickCoords.x;

    surf3Dwrite(make_uchar4(tmp.x,tmp.y,tmp.z,tmp.w), surfRef, newCoords.x*sizeof(uchar4), newCoords.y, newCoords.z);

    // top side: 0, 1, 4, 5
    tmp = make_float4(0,0,0,0);
    tmp.x += static_cast<unsigned int>(colors[0].x);
    tmp.y += static_cast<unsigned int>(colors[0].y);
    tmp.z += static_cast<unsigned int>(colors[0].z);
    tmp.w += static_cast<unsigned int>(colors[0].w);

    tmp.x += static_cast<unsigned int>(colors[1].x);
    tmp.y += static_cast<unsigned int>(colors[1].y);
    tmp.z += static_cast<unsigned int>(colors[1].z);
    tmp.w += static_cast<unsigned int>(colors[1].w);

    tmp.x += static_cast<unsigned int>(colors[4].x);
    tmp.y += static_cast<unsigned int>(colors[4].y);
    tmp.z += static_cast<unsigned int>(colors[4].z);
    tmp.w += static_cast<unsigned int>(colors[4].w);

    tmp.x += static_cast<unsigned int>(colors[5].x);
    tmp.y += static_cast<unsigned int>(colors[5].y);
    tmp.z += static_cast<unsigned int>(colors[5].z);
    tmp.w += static_cast<unsigned int>(colors[5].w);

    tmp.x *= 0.25f;
    tmp.y *= 0.25f;
    tmp.z *= 0.25f;
    tmp.w *= 0.25f;

    newCoords = make_uint3(1,0,1);
    newCoords.x+=brickCoords.x;
    newCoords.y+=brickCoords.x;
    newCoords.z+=brickCoords.x;

    surf3Dwrite(make_uchar4(tmp.x,tmp.y,tmp.z,tmp.w), surfRef, newCoords.x*sizeof(uchar4), newCoords.y, newCoords.z);

    // near side: 0, 1, 2, 3
    tmp = make_float4(0,0,0,0);
    tmp.x += static_cast<unsigned int>(colors[0].x);
    tmp.y += static_cast<unsigned int>(colors[0].y);
    tmp.z += static_cast<unsigned int>(colors[0].z);
    tmp.w += static_cast<unsigned int>(colors[0].w);

    tmp.x += static_cast<unsigned int>(colors[1].x);
    tmp.y += static_cast<unsigned int>(colors[1].y);
    tmp.z += static_cast<unsigned int>(colors[1].z);
    tmp.w += static_cast<unsigned int>(colors[1].w);

    tmp.x += static_cast<unsigned int>(colors[2].x);
    tmp.y += static_cast<unsigned int>(colors[2].y);
    tmp.z += static_cast<unsigned int>(colors[2].z);
    tmp.w += static_cast<unsigned int>(colors[2].w);

    tmp.x += static_cast<unsigned int>(colors[3].x);
    tmp.y += static_cast<unsigned int>(colors[3].y);
    tmp.z += static_cast<unsigned int>(colors[3].z);
    tmp.w += static_cast<unsigned int>(colors[3].w);

    tmp.x *= 0.25f;
    tmp.y *= 0.25f;
    tmp.z *= 0.25f;
    tmp.w *= 0.25f;

    newCoords = make_uint3(1,1,0);
    newCoords.x+=brickCoords.x;
    newCoords.y+=brickCoords.x;
    newCoords.z+=brickCoords.x;

    surf3Dwrite(make_uchar4(tmp.x,tmp.y,tmp.z,tmp.w), surfRef, newCoords.x*sizeof(uchar4), newCoords.y, newCoords.z);

    // far side: 4, 5, 6, 7
    tmp = make_float4(0,0,0,0);
    tmp.x += static_cast<unsigned int>(colors[4].x);
    tmp.y += static_cast<unsigned int>(colors[4].y);
    tmp.z += static_cast<unsigned int>(colors[4].z);
    tmp.w += static_cast<unsigned int>(colors[4].w);

    tmp.x += static_cast<unsigned int>(colors[5].x);
    tmp.y += static_cast<unsigned int>(colors[5].y);
    tmp.z += static_cast<unsigned int>(colors[5].z);
    tmp.w += static_cast<unsigned int>(colors[5].w);

    tmp.x += static_cast<unsigned int>(colors[6].x);
    tmp.y += static_cast<unsigned int>(colors[6].y);
    tmp.z += static_cast<unsigned int>(colors[6].z);
    tmp.w += static_cast<unsigned int>(colors[6].w);

    tmp.x += static_cast<unsigned int>(colors[7].x);
    tmp.y += static_cast<unsigned int>(colors[7].y);
    tmp.z += static_cast<unsigned int>(colors[7].z);
    tmp.w += static_cast<unsigned int>(colors[7].w);

    tmp.x *= 0.25f;
    tmp.y *= 0.25f;
    tmp.z *= 0.25f;
    tmp.w *= 0.25f;

    newCoords = make_uint3(1,1,2);
    newCoords.x+=brickCoords.x;
    newCoords.y+=brickCoords.x;
    newCoords.z+=brickCoords.x;

    surf3Dwrite(make_uchar4(tmp.x,tmp.y,tmp.z,tmp.w), surfRef, newCoords.x*sizeof(uchar4), newCoords.y, newCoords.z);

    // ####################### EDGES (FRONT) #####################
    // top edge
    tmp = make_float4(0,0,0,0);
    tmp.x += static_cast<unsigned int>(colors[0].x);
    tmp.y += static_cast<unsigned int>(colors[0].y);
    tmp.z += static_cast<unsigned int>(colors[0].z);
    tmp.w += static_cast<unsigned int>(colors[0].w);

    tmp.x += static_cast<unsigned int>(colors[1].x);
    tmp.y += static_cast<unsigned int>(colors[1].y);
    tmp.z += static_cast<unsigned int>(colors[1].z);
    tmp.w += static_cast<unsigned int>(colors[1].w);

    tmp.x *= 0.5f;
    tmp.y *= 0.5f;
    tmp.z *= 0.5f;
    tmp.w *= 0.5f;

    newCoords = make_uint3(1,0,0);
    newCoords.x+=brickCoords.x;
    newCoords.y+=brickCoords.x;
    newCoords.z+=brickCoords.x;

    surf3Dwrite(make_uchar4(tmp.x,tmp.y,tmp.z,tmp.w), surfRef, newCoords.x*sizeof(uchar4), newCoords.y, newCoords.z);

    // bottom edge
    tmp = make_float4(0,0,0,0);
    tmp.x += static_cast<unsigned int>(colors[2].x);
    tmp.y += static_cast<unsigned int>(colors[2].y);
    tmp.z += static_cast<unsigned int>(colors[2].z);
    tmp.w += static_cast<unsigned int>(colors[2].w);

    tmp.x += static_cast<unsigned int>(colors[3].x);
    tmp.y += static_cast<unsigned int>(colors[3].y);
    tmp.z += static_cast<unsigned int>(colors[3].z);
    tmp.w += static_cast<unsigned int>(colors[3].w);

    tmp.x *= 0.5f;
    tmp.y *= 0.5f;
    tmp.z *= 0.5f;
    tmp.w *= 0.5f;

    newCoords = make_uint3(1,2,0);
    newCoords.x+=brickCoords.x;
    newCoords.y+=brickCoords.x;
    newCoords.z+=brickCoords.x;

    surf3Dwrite(make_uchar4(tmp.x,tmp.y,tmp.z,tmp.w), surfRef, newCoords.x*sizeof(uchar4), newCoords.y, newCoords.z);

    // left edge
    tmp = make_float4(0,0,0,0);
    tmp.x += static_cast<unsigned int>(colors[0].x);
    tmp.y += static_cast<unsigned int>(colors[0].y);
    tmp.z += static_cast<unsigned int>(colors[0].z);
    tmp.w += static_cast<unsigned int>(colors[0].w);

    tmp.x += static_cast<unsigned int>(colors[3].x);
    tmp.y += static_cast<unsigned int>(colors[3].y);
    tmp.z += static_cast<unsigned int>(colors[3].z);
    tmp.w += static_cast<unsigned int>(colors[3].w);

    tmp.x *= 0.5f;
    tmp.y *= 0.5f;
    tmp.z *= 0.5f;
    tmp.w *= 0.5f;

    newCoords = make_uint3(0,1,0);
    newCoords.x+=brickCoords.x;
    newCoords.y+=brickCoords.x;
    newCoords.z+=brickCoords.x;

    surf3Dwrite(make_uchar4(tmp.x,tmp.y,tmp.z,tmp.w), surfRef, newCoords.x*sizeof(uchar4), newCoords.y, newCoords.z);

    // right edge
    tmp = make_float4(0,0,0,0);
    tmp.x += static_cast<unsigned int>(colors[1].x);
    tmp.y += static_cast<unsigned int>(colors[1].y);
    tmp.z += static_cast<unsigned int>(colors[1].z);
    tmp.w += static_cast<unsigned int>(colors[1].w);

    tmp.x += static_cast<unsigned int>(colors[2].x);
    tmp.y += static_cast<unsigned int>(colors[2].y);
    tmp.z += static_cast<unsigned int>(colors[2].z);
    tmp.w += static_cast<unsigned int>(colors[2].w);

    tmp.x *= 0.5f;
    tmp.y *= 0.5f;
    tmp.z *= 0.5f;
    tmp.w *= 0.5f;

    newCoords = make_uint3(2,1,0);
    newCoords.x+=brickCoords.x;
    newCoords.y+=brickCoords.x;
    newCoords.z+=brickCoords.x;

    surf3Dwrite(make_uchar4(tmp.x,tmp.y,tmp.z,tmp.w), surfRef, newCoords.x*sizeof(uchar4), newCoords.y, newCoords.z);

    // ####################### EDGES (BACK) #####################
    // top edge
    tmp = make_float4(0,0,0,0);
    tmp.x += static_cast<unsigned int>(colors[4].x);
    tmp.y += static_cast<unsigned int>(colors[4].y);
    tmp.z += static_cast<unsigned int>(colors[4].z);
    tmp.w += static_cast<unsigned int>(colors[4].w);

    tmp.x += static_cast<unsigned int>(colors[5].x);
    tmp.y += static_cast<unsigned int>(colors[5].y);
    tmp.z += static_cast<unsigned int>(colors[5].z);
    tmp.w += static_cast<unsigned int>(colors[5].w);

    tmp.x *= 0.5f;
    tmp.y *= 0.5f;
    tmp.z *= 0.5f;
    tmp.w *= 0.5f;

    newCoords = make_uint3(1,0,2);
    newCoords.x+=brickCoords.x;
    newCoords.y+=brickCoords.x;
    newCoords.z+=brickCoords.x;

    surf3Dwrite(make_uchar4(tmp.x,tmp.y,tmp.z,tmp.w), surfRef, newCoords.x*sizeof(uchar4), newCoords.y, newCoords.z);

    // bottom edge
    tmp = make_float4(0,0,0,0);
    tmp.x += static_cast<unsigned int>(colors[6].x);
    tmp.y += static_cast<unsigned int>(colors[6].y);
    tmp.z += static_cast<unsigned int>(colors[6].z);
    tmp.w += static_cast<unsigned int>(colors[6].w);

    tmp.x += static_cast<unsigned int>(colors[7].x);
    tmp.y += static_cast<unsigned int>(colors[7].y);
    tmp.z += static_cast<unsigned int>(colors[7].z);
    tmp.w += static_cast<unsigned int>(colors[7].w);

    tmp.x *= 0.5f;
    tmp.y *= 0.5f;
    tmp.z *= 0.5f;
    tmp.w *= 0.5f;

    newCoords = make_uint3(1,2,2);
    newCoords.x+=brickCoords.x;
    newCoords.y+=brickCoords.x;
    newCoords.z+=brickCoords.x;

    surf3Dwrite(make_uchar4(tmp.x,tmp.y,tmp.z,tmp.w), surfRef, newCoords.x*sizeof(uchar4), newCoords.y, newCoords.z);

    // left edge
    tmp = make_float4(0,0,0,0);
    tmp.x += static_cast<unsigned int>(colors[4].x);
    tmp.y += static_cast<unsigned int>(colors[4].y);
    tmp.z += static_cast<unsigned int>(colors[4].z);
    tmp.w += static_cast<unsigned int>(colors[4].w);

    tmp.x += static_cast<unsigned int>(colors[7].x);
    tmp.y += static_cast<unsigned int>(colors[7].y);
    tmp.z += static_cast<unsigned int>(colors[7].z);
    tmp.w += static_cast<unsigned int>(colors[7].w);

    tmp.x *= 0.5f;
    tmp.y *= 0.5f;
    tmp.z *= 0.5f;
    tmp.w *= 0.5f;

    newCoords = make_uint3(0,1,2);
    newCoords.x+=brickCoords.x;
    newCoords.y+=brickCoords.x;
    newCoords.z+=brickCoords.x;

    surf3Dwrite(make_uchar4(tmp.x,tmp.y,tmp.z,tmp.w), surfRef, newCoords.x*sizeof(uchar4), newCoords.y, newCoords.z);

    // right edge
    tmp = make_float4(0,0,0,0);
    tmp.x += static_cast<unsigned int>(colors[5].x);
    tmp.y += static_cast<unsigned int>(colors[5].y);
    tmp.z += static_cast<unsigned int>(colors[5].z);
    tmp.w += static_cast<unsigned int>(colors[5].w);

    tmp.x += static_cast<unsigned int>(colors[6].x);
    tmp.y += static_cast<unsigned int>(colors[6].y);
    tmp.z += static_cast<unsigned int>(colors[6].z);
    tmp.w += static_cast<unsigned int>(colors[6].w);

    tmp.x *= 0.5f;
    tmp.y *= 0.5f;
    tmp.z *= 0.5f;
    tmp.w *= 0.5f;

    newCoords = make_uint3(2,1,2);
    newCoords.x+=brickCoords.x;
    newCoords.y+=brickCoords.x;
    newCoords.z+=brickCoords.x;

    surf3Dwrite(make_uchar4(tmp.x,tmp.y,tmp.z,tmp.w), surfRef, newCoords.x*sizeof(uchar4), newCoords.y, newCoords.z);
}

#endif