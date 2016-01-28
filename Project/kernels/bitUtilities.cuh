#ifndef BIT_UTILITIES_CUH
#define BIT_UTILITIES_CUH

// returns a bit from value at the specified position
__device__
unsigned int getBit(unsigned int value, unsigned int position)
{
    return (value >> (position-1)) & 1u;
}

// sets a bit from value at the specified position
__device__
void setBit(unsigned int &value, unsigned int position)
{
    value |= (1u << (position-1));
}

// unsets a bit from value at the specified position (bit value will be 0)
__device__
void unSetBit(unsigned int &value, unsigned int position)
{
    value &= ~(1u << (position-1));
}

__device__
void getVoxelPositionUINTtoFLOAT3(const unsigned int codedPosition, float3 &position)
{
    // mask to get 10 bit position coords
    const unsigned int mask_bits = 0x000003FF;

    // dont forget the .f for casting reasons :P
    position.x = ((codedPosition) & (mask_bits)) / 1024.f;
    position.y = ((codedPosition >> 10) & (mask_bits)) / 1024.f;
    position.z = ((codedPosition >> 20) & (mask_bits)) / 1024.f;
}

#endif