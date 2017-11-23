

#include "ImplicitSolver.hpp"
#include "Domain.hpp"
#include "Log.hpp"

#include "ModuleList.hpp"
#include "System.hpp"
#include "RootFinder.hpp"
#include "Periodic.hpp"

REGISTERIMPL(ImplicitSolver);

std::string ImplicitSolver::moduleName() const
{
  return "ImplicitSolver";
}

void ImplicitSolver::initialise(const real alpha)
{
  m_alpha = alpha;
}

void ImplicitSolver::initialiseFromParameters(const Parameters& params)
{
  this->initialise(
      getParameter<real>(params, "alpha",0.5)
  );
}


//! Function which needs to be 0
/**
 * This is the function we need to find the root of. If f==0 Then the array states_new that
 * was passed to the function solves the PDE
 *
 * states_old [in] Concatenated array of states at t^n
 * states_new [in] Concatenated array of states  t^n+1
 * f [out] Array which should be 0
 */
void ImplicitSolver::function(const Vector& states_old, const Vector& states_new, const real dt, const real t, Vector& f) const
{
  const std::shared_ptr<const System> system = ModuleList::uniqueModule<System>();
  const std::shared_ptr<const Domain> domain = ModuleList::uniqueModule<Domain>();

  assert(f.size() == domain->cells()*SystemAttributes::stateSize);
  assert(states_old.size() == states_new.size());
  assert(states_old.size() == f.size());
  assert(m_bcs); // Check if boundary conditions were set

  const real ds = domain->ds();

  const State factorDt = system->factorTimeDeriv();

  const std::shared_ptr<const BoundaryCondition> bcL = m_bcs->left();
  const std::shared_ptr<const BoundaryCondition> bcR = m_bcs->right();

  const int stenS = SystemAttributes::stencilSize;
  const int statS = SystemAttributes::stateSize;

  const int end = domain->end();

  for(int i=domain->begin();i<domain->end();i++)
  {
    const real s = domain->s(i);

    StencilArray stencil_old;
    StencilArray stencil_new;
    for(int k=i-stenS,j=0;k<=i+stenS;k++,j++)
    {
      if(k<domain->begin())
      {
        // Boundary condition required on the left edge
        // Make it transmissive
        State domainState_old;
        State domainState_new;
        State periodicDomainState_old;
        State periodicDomainState_new;
        for(unsigned int u=0;u<statS;u++)
        {
          domainState_new[u] = states_new[(-k-1)*statS+u];
          domainState_old[u] = states_old[(-k-1)*statS+u];
          periodicDomainState_new[u] = states_new[(end+k)*statS+u];
          periodicDomainState_old[u] = states_old[(end+k)*statS+u];
        }
        stencil_new[j] = bcL->ghostState(domainState_new,periodicDomainState_new);
        stencil_old[j] = bcL->ghostState(domainState_old,periodicDomainState_old);
      }
      else if(domain->end()<=k)
      {
        // Boundary condition required on the right edge
        // Make it transmissive
        State domainState_old;
        State domainState_new;
        State periodicDomainState_old;
        State periodicDomainState_new;
        for(unsigned int u=0;u<statS;u++)
        {
          domainState_new[u] = states_new[statS*(2*domain->end()-k-1)+u];
          domainState_old[u] = states_old[statS*(2*domain->end()-k-1)+u];
          periodicDomainState_new[u] = states_new[(k-end)*statS+u];
          periodicDomainState_old[u] = states_old[(k-end)*statS+u];
        }
        stencil_new[j] = bcR->ghostState(domainState_new,periodicDomainState_new);
        stencil_old[j] = bcR->ghostState(domainState_old,periodicDomainState_old);
      }
      else
      {
        for(unsigned int u=0;u<statS;u++)
        {
          stencil_old[j][u] = states_old[statS*k+u];
          stencil_new[j][u] = states_new[statS*k+u];
        }
      }
    }
    const State F_old = system->F(stencil_old,ds,s,t);
    const State F_new = system->F(stencil_new,ds,s,t);
    

    for(unsigned int e=0;e<SystemAttributes::stateSize;e++)
    {
      f[i*statS+e] = factorDt[e]*(states_new[i*statS+e] - states_old[i*statS+e]) + dt*(m_alpha*F_new[e] + (1.-m_alpha)*F_old[e]);
    }
  }
}

//!
/**
 *
 * J_{i j} = \partial
 *
 * J [out] - Jacobian of function
 */
void ImplicitSolver::jacobian(const Vector& states, const real dt, const real t, SpMatRowMaj& J) const
{
  const int stenS = SystemAttributes::stencilSize;
  const int statS = SystemAttributes::stateSize;

  const std::shared_ptr<const System> system = ModuleList::uniqueModule<System>();
  const std::shared_ptr<const Domain> domain = ModuleList::uniqueModule<Domain>();

  assert(m_bcs); // Check if boundary conditions were set
  const std::shared_ptr<const BoundaryCondition> bcL = m_bcs->left();
  const std::shared_ptr<const BoundaryCondition> bcR = m_bcs->right();

  // TODO Find a better way to incorporate periodic boundary conditions
  const bool isPeriodicL = std::dynamic_pointer_cast<const Periodic>(bcL).use_count() > 0;
  const bool isPeriodicR = std::dynamic_pointer_cast<const Periodic>(bcR).use_count() > 0;

  assert((unsigned int)(J.rows()) == domain->cells()*statS);
  assert((unsigned int)(J.cols()) == domain->cells()*statS);
  assert((unsigned int)(states.size()) == domain->cells()*statS);

  const real ds = domain->ds();
  const State factorDt = system->factorTimeDeriv();

  const int end = domain->end();

  typedef Eigen::Triplet<real> Triplet;
  std::vector<Triplet> triplets;
  // Size of stencil jacobian * number of cells
  triplets.reserve(statS*(stenS*2+1)*domain->cells());

  for(int cell=domain->begin();cell<domain->end();cell++)
  {
    const real s = domain->s(cell);

    StencilArray stencil;

#ifdef DEBUG
    // Initialise to nan so we can be sure the stencil is set properly
    for(int i=0;i<2*stenS+1;i++)
    {
      for(int j=0;j<statS;j++)
      {
        stencil[i][j] = -std::numeric_limits<real>::quiet_NaN();
      }
    }
#endif

    for(int i_stenc_cell=cell-stenS,i_stenc=0;i_stenc_cell<=cell+stenS;i_stenc_cell++,i_stenc++)
    {
      // i_stenc iterates over the stencil = 0,1,...,stencilSize
      // i_stenc_cell iterates over the cells covered by the stencil
      /*
       *       0  |  1  |  2  |  3  |  4  | ...
       *             '-----------'
       *               stencil
       *
       * Then i_stenc_cell = 1,2,3
       */
      if(i_stenc_cell<domain->begin())
      {
        // Boundary condition required on the left edge
        // Make it transmissive
        State domainState, periodicDomainState;
        for(unsigned int i_state=0;i_state<statS;i_state++)
        {
          domainState[i_state] = states[(-i_stenc_cell-1)*statS+i_state];
          periodicDomainState[i_state] = states[(end+i_stenc_cell)*statS+i_state];
        }
        stencil[i_stenc] = bcL->ghostState(domainState,periodicDomainState);
      }
      else if(domain->end()<=i_stenc_cell)
      {
        // Boundary condition required on the right edge
        // Make it transmissive
        State domainState, periodicDomainState;
        for(unsigned int i_state=0;i_state<statS;i_state++)
        {
          domainState[i_state] = states[statS*(2*domain->end()-i_stenc_cell-1)+i_state];
          periodicDomainState[i_state] = states[(i_stenc_cell-end)*statS+i_state];
        }
        stencil[i_stenc] = bcR->ghostState(domainState,periodicDomainState);
      }
      else
      {
        for(unsigned int i_state=0;i_state<statS;i_state++)
        {
          stencil[i_stenc][i_state] = states[statS*i_stenc_cell+i_state];
        }
      }
    }

    const StencilJacobian J_loc = system->J(stencil,ds,s,t);

    // J_loc is column major so iterate over rows faster
    for(unsigned int J_loc_col=0;J_loc_col<statS*(stenS*2+1);J_loc_col++)
    {
      int i_stenc = cell*statS - (signed int)(stenS*statS) + (signed int)(J_loc_col);
      int i_stenc_cell = cell-(signed int)(stenS)+(signed int)(J_loc_col)/(signed int)(statS);

      if(isPeriodicL && i_stenc_cell < 0)
      {
        i_stenc = (i_stenc+domain->end()*statS) % (domain->end()*statS);
        i_stenc_cell = (i_stenc_cell + domain->end()) % domain->end();
      }
      if(isPeriodicR && domain->end() <= i_stenc_cell )
      {
        i_stenc = i_stenc % (domain->end()*statS);
        i_stenc_cell = i_stenc_cell % domain->end();
      }

      if(domain->begin() <= i_stenc_cell && 
                            i_stenc_cell < domain->end())
      {
        for(unsigned int i_state=0;i_state<statS;i_state++)
        {
          real deriv = dt*m_alpha*J_loc(i_state,J_loc_col);
          const int J_col = i_stenc;
          const int J_row = cell*statS+i_state;
          if(J_col == J_row)
          {
            deriv += factorDt[i_state]; // Take into account that the state appears in the discr time deriv
          }
          const Triplet trip = Triplet(J_row,J_col,deriv);
          triplets.push_back(trip);
        }
      }
    }
  }
  J.setZero();
  J.setFromTriplets(triplets.begin(),triplets.end());
}

void ImplicitSolver::advance(std::shared_ptr<DataPatch> states, const real dt, const real t) const
{
  BOOST_LOG_TRIVIAL(debug) << "ImplicitSolver: Advancing data patch by dt = " << dt << ", t = " << t;

  const std::shared_ptr<const RootFinder> solver = ModuleList::uniqueModule<RootFinder>();

  const std::shared_ptr<const BoundaryCondition> bcL = m_bcs->left();
  const std::shared_ptr<const BoundaryCondition> bcR = m_bcs->right();

  // IMPROVE this by using Eigen::map and not copy all the states
  const unsigned int statS = SystemAttributes::stateSize;
  Vector states_vec_new(states->rows()*statS);
  Vector states_vec_old(states->rows()*statS);

  real c_t = 0.;
  real c_dt = dt;
  do
  {
    c_dt = std::min(dt - c_t, c_dt); 

    // Copy states to states_vec_old
    for(unsigned int i=0;i<states->rows();i++)
    {
      for(unsigned int j=0;j<statS;j++)
      {
        // Can't use Eigen::map here since states is column major. We could change the ordering
        // but this is an intricate refactoring and probably not worth it at this point
        states_vec_old[i*statS+j] = (*states)(i,j);
        states_vec_new[i*statS+j] = (*states)(i,j);
      }
    }

    // f and J should have the signature NetwonRaphson's solveSparse accepts
    auto f = std::bind(&ImplicitSolver::function,*this,states_vec_old,std::placeholders::_1,c_dt,t+c_t,std::placeholders::_2);
    auto J = std::bind(&ImplicitSolver::jacobian,*this,std::placeholders::_1,c_dt,t+c_t,std::placeholders::_2);

    // Restrict the search domain to valid states
    auto restrictDomain = std::bind(&ImplicitSolver::checkValid, *this, std::placeholders::_1);

    bool converged = false;
    try
    {
      // Solve system
      solver->solveSparse(f, J, states_vec_new, restrictDomain);
      converged = true;
    }
    catch(RootFinder::ConvergenceException e)
    {
      converged = false;
    }

    if (converged)
    {
      // Copy states_vec_new to states
      for(unsigned int i=0;i<states->rows();i++)
      {
        for(unsigned int j=0;j<statS;j++)
        {
          (*states)(i,j) = states_vec_new[i*statS+j];
        }
      }

      c_t += c_dt;
    }
    else
    {
      c_dt *= 0.5; 
      BOOST_LOG_TRIVIAL(error) << "Time step did not converge. Trying smaller dt = " << c_dt;
    }

  } while (c_t < dt);
}


bool ImplicitSolver::checkValid(Vector& states) const
{
  const int statS = SystemAttributes::stateSize;
  const std::shared_ptr<const System> system = ModuleList::uniqueModule<System>();

  assert((states.size() % statS) == 0);
  
  const int numCells = states.size()/statS;

  if(!states.allFinite())
  {
    BOOST_LOG_TRIVIAL(error) << "Non-finite states found. It cannot be recovered from these states.";
    return false;
  }

  for(int cell=0;cell<numCells;cell++)
  {
    // Do this as a map to avoid copying
    Eigen::Map<State> state = Eigen::Map<State>(&(states[statS*cell]));

    if(!system->checkValid(state))
    {
      return false;
    }
  }

  return true;
}

