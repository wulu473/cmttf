
#ifndef FLAT_H_
#define FLAT_H_

#include "Domain.hpp"

class Flat : public Domain
{
  REGISTER(Flat);
  public:
    virtual std::string moduleName() const;
    Flat();
    virtual ~Flat();

    virtual real x(const int) const;
    virtual real dx() const;
    virtual real minX() const; 
    virtual real maxX() const;
    virtual int begin() const;
    virtual int end() const;
    virtual unsigned int cells() const;

    virtual void initialiseFromFile();
    void initialise(const int N, const real L, const real R);
  protected:
  private:
    int m_iL, m_iR, m_N;;
    real m_xL, m_xR, m_dx;
};

#endif
