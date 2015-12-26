//
// Created by nils1990 on 04.12.15.
//

#ifndef CUDAEXCEPTION_H
#define CUDAEXCEPTION_H

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


#endif //CUDAEXCEPTION_H
