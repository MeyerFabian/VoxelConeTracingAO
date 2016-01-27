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


#endif