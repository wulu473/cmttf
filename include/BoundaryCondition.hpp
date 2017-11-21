
#ifndef BOUNDARYCONDITION_H_
#define BOUNDARYCONDITION_H_

#include "Parameters.hpp"
#include "ModuleBase.hpp"

class BoundaryCondition : public ParameterisedModuleBase
{
  public:
    // Define choices where this boundary condition applies
    enum Orientation {Left, Right};

    virtual std::string baseName() const final;

    //! Set the orientation of this boundary condition
    virtual void setOrientation(const Orientation o);

    //! Return the state of the ghost cell
    virtual State ghostState(const State& domainState, const State& periodicDomainState) const;

  protected:
    //! Enum whether this is the left or the right boudnary condition
    Orientation m_orientation;
};

class BoundaryConditionContainer
{
  public:
    BoundaryConditionContainer();

    void initialise(const std::shared_ptr<BoundaryCondition> left,
                    const std::shared_ptr<BoundaryCondition> right);
    void initialiseFromParameters(const Parameters& params);
    
    std::shared_ptr<const BoundaryCondition> left() const;
    std::shared_ptr<const BoundaryCondition> right() const;

  private:
    std::shared_ptr<BoundaryCondition> m_left;
    std::shared_ptr<BoundaryCondition> m_right;
};

#endif

