
#ifndef DOMAIN_H_
#define DOMAIN_H_

#include "ModuleBase.hpp"
#include "Attributes.hpp"

class Domain : public ParameterisedModuleBase
{
  public:
    virtual std::string baseName() const final;

    virtual void initialise();

    //! Return coordinate in embedding space
    virtual Coord x(const real s) const;

    //! i-th basis vector
    virtual Coord x_(const unsigned int i, const real s) const;

    //! Return surface coordinate of cell
    virtual real s(const int i) const;

    //! Return surface curvature
    virtual real kappa(const real s) const;

    //! Return derivative of surface curvature
    virtual real dkappa_ds(const real s) const;

    //! Return spacing along surface
    virtual real ds() const ;

    //! Return lower bound of domain on surface
    virtual real minS() const ; 

    //! Return upper bound of domain on surface
    virtual real maxS() const ;

    //! Return index of first cell
    virtual int begin() const;

    //! Return last valid index + 1
    virtual int end() const;

    //! Return number of cells
    virtual unsigned int cells() const;

    Domain();
    virtual ~Domain();

  protected:
    //! number of cells
    unsigned int m_nCells;

    //! bounds of surface coords
    real m_minS, m_maxS;
};
#endif 
