#include "CudaException.h"

CudaException::CudaException(std::string errorDescription)
{
    m_errorDescription = errorDescription;
}

const char *CudaException::what() const throw()
{
    return m_errorDescription.c_str();
}

CudaException::~CudaException()
{
    // Nothing to do
}
