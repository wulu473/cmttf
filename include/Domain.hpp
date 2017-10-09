
#ifndef DOMAIN_H_
#define DOMAIN_H_

#include "ModuleBase.hpp"
#include "Attributes.hpp"

class Domain : public ParameterisedModuleBase
{
  public:
    virtual std::string baseName() const final;

    virtual void initialise();

    virtual real x(const int) const ;
    virtual real dx() const ;
    virtual real minX() const ; 
    virtual real maxX() const ;
    virtual int begin() const ;
    virtual int end() const;
    virtual unsigned int cells() const;

    Domain();
    virtual ~Domain();

  protected:
};
#endif 
