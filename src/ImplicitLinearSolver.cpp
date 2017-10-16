

#include "ImplicitLinearSolver.hpp"
#include "Domain.hpp"

#include "ModuleList.hpp"
#include "System.hpp"
#include "RootFinder.hpp"

REGISTERIMPL(ImplicitLinearSolver);

std::string ImplicitLinearSolver::moduleName() const
{
  return "ImplicitLinearSolver";
}

void ImplicitLinearSolver::initialise(const real alpha)
{
  m_alpha = alpha;
}

void ImplicitLinearSolver::initialiseFromFile()
{
  this->initialise(
      getParameter<real>("alpha",0.5)
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
void ImplicitLinearSolver::function(const Vector& states_old, const Vector& states_new, const real dt, const real t, Vector& f) const
{
  static const std::shared_ptr<const System> system = Modules::uniqueModule<System>();
  static const std::shared_ptr<const Domain> domain = Modules::uniqueModule<Domain>();

  assert(f.size() == domain->cells()*SystemAttributes::stateSize);
  assert(states_old.size() == states_new.size());
  assert(states_old.size() == f.size());

  const real dx = domain->dx();

  const int stenS = SystemAttributes::stencilSize;
  const int statS = SystemAttributes::stateSize;

  for(int i=domain->begin();i<domain->end();i++)
  {
    const real x = domain->x(i);

    StencilArray stencil_old;
    StencilArray stencil_new;
    for(int k=i-stenS,j=0;k<=i+stenS;k++,j++)
    {
      if(k<domain->begin())
      {
        // Boundary condition required on the left edge
        // Make it transmissive
        for(unsigned int u=0;u<statS;u++)
        {
          stencil_new[j][u] = states_new[(-k-1)*statS+u];
          stencil_old[j][u] = states_old[(-k-1)*statS+u];
        }
      }
      else if(domain->end()<=k)
      {
        // Boundary condition required on the right edge
        // Make it transmissive
        for(unsigned int u=0;u<statS;u++)
        {
          stencil_new[j][u] = states_new[statS*(2*domain->end()-k-1)+u];
          stencil_old[j][u] = states_old[statS*(2*domain->end()-k-1)+u];
        }
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
    const State F_old = system->F(stencil_old,dx,x,t);
    const State F_new = system->FLinear(stencil_new,stencil_old,dx,x,t);
    

    for(unsigned int e=0;e<SystemAttributes::stateSize;e++)
    {
      f[i*statS+e] = states_new[i*statS+e] - states_old[i*statS+e] + dt*(m_alpha*F_new[e] + (1.-m_alpha)*F_old[e]);
    }
  }
}

void ImplicitLinearSolver::jacobian(const Vector& states_old, const Vector& states_new, const real dt, const real t, SpMatRowMaj& J) const
{
  const int stenS = SystemAttributes::stencilSize;
  const int statS = SystemAttributes::stateSize;

  const static std::shared_ptr<const System> system = Modules::uniqueModule<System>();
  const static std::shared_ptr<const Domain> domain = Modules::uniqueModule<Domain>();

  assert((unsigned int)(J.rows()) == domain->cells()*statS);
  assert((unsigned int)(J.cols()) == domain->cells()*statS);
  assert((unsigned int)(states_new.size()) == domain->cells()*statS);

  const real dx = domain->dx();

  typedef Eigen::Triplet<real> Triplet;
  std::vector<Triplet> triplets;
  // Size of stencil jacobian * number of cells
  triplets.reserve(statS*(stenS*2+1)*domain->cells());

  for(int cell=domain->begin();cell<domain->end();cell++)
  {
    const real x = domain->x(cell);

    StencilArray stencil_old, stencil_new;

#ifdef DEBUG
    // Initialise to nan so we can be sure the stencil is set properly
    for(int i=0;i<2*stenS+1;i++)
    {
      for(int j=0;j<statS;j++)
      {
        stencil_old[i][j] = -std::numeric_limits<real>::quiet_NaN();
        stencil_new[i][j] = -std::numeric_limits<real>::quiet_NaN();
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
        for(unsigned int i_state=0;i_state<statS;i_state++)
        {
          stencil_old[i_stenc][i_state] = states_old[(-i_stenc_cell-1)*statS+i_state];
          stencil_new[i_stenc][i_state] = states_new[(-i_stenc_cell-1)*statS+i_state];
        }
      }
      else if(domain->end()<=i_stenc_cell)
      {
        // Boundary condition required on the right edge
        // Make it transmissive
        for(unsigned int i_state=0;i_state<statS;i_state++)
        {
          stencil_old[i_stenc][i_state] = states_old[statS*(2*domain->end()-i_stenc_cell-1)+i_state];
          stencil_new[i_stenc][i_state] = states_new[statS*(2*domain->end()-i_stenc_cell-1)+i_state];
        }
      }
      else
      {
        for(unsigned int i_state=0;i_state<statS;i_state++)
        {
          stencil_old[i_stenc][i_state] = states_old[statS*i_stenc_cell+i_state];
          stencil_new[i_stenc][i_state] = states_new[statS*i_stenc_cell+i_state];
        }
      }
    }

    const StencilJacobian J_loc = system->JLinear(stencil_new,stencil_old,dx,x,t);

    // J_loc is column major so iterate over rows faster
    for(unsigned int J_loc_col=0;J_loc_col<statS*(stenS*2+1);J_loc_col++)
    {
      const int i_stenc = cell*statS - (signed int)(stenS*statS) + (signed int)(J_loc_col);
      const int i_stenc_cell = cell-(signed int)(stenS)+(signed int)(J_loc_col)/(signed int)(statS);
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
            deriv += 1.0; // Take into account that the state appears in the discr time deriv
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

void ImplicitLinearSolver::advance(std::shared_ptr<DataPatch> states, const real dt, const real t) const
{
  BOOST_LOG_TRIVIAL(debug) << "ImplicitLinearSolver: Advancing data patch by dt = " << dt << ", t = " << t;

  static const std::shared_ptr<const RootFinder> solver = Modules::uniqueModule<RootFinder>();

  // IMPROVE this by using Eigen::map and not copy all the states
  const unsigned int statS = SystemAttributes::stateSize;
  Vector states_vec_new(states->rows()*statS);
  Vector states_vec_old(states->rows()*statS);


  // Copy states to states_vec_old
  for(unsigned int i=0;i<states->rows();i++)
  {
    for(unsigned int j=0;j<statS;j++)
    {
      states_vec_old[i*statS+j] = (*states)(i,j);
      states_vec_new[i*statS+j] = (*states)(i,j);
    }
  }

  // f and J should have the signature NetwonRaphson's solveSparse accepts
  auto f = std::bind(&ImplicitLinearSolver::function,*this,states_vec_old,std::placeholders::_1,dt,t,std::placeholders::_2);
  auto J = std::bind(&ImplicitLinearSolver::jacobian,*this,states_vec_old,std::placeholders::_1,dt,t,std::placeholders::_2);

  // Solve system
  solver->solveSparse(f,J,states_vec_new);
  // Copy states_vec_new to states
  for(unsigned int i=0;i<states->rows();i++)
  {
    for(unsigned int j=0;j<statS;j++)
    {
      (*states)(i,j) = states_vec_new[i*statS+j];
    }
  }
}


