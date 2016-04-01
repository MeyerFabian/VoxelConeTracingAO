#ifndef CUDA_EXCEPTION_H
#define CUDA_EXCEPTION_H

#include <exception>
#include <string>

class CudaException : public std::exception
{
public:

    CudaException(std::string errorDescription);
    virtual ~CudaException();
    virtual const char *what() const throw();

protected:

    std::string m_errorDescription;
};


#endif //CUDA_EXCEPTION_H
