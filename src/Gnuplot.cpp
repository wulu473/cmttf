
#include <fstream>
#include <sstream>
#include <iomanip>

#include "ModuleList.hpp"
#include "Domain.hpp"
#include "Git.hpp"
#include "System.hpp"

#include "Gnuplot.hpp"

REGISTERIMPL(Gnuplot);

Gnuplot::Gnuplot()
{

}

Gnuplot::~Gnuplot()
{

}

std::string Gnuplot::moduleName() const
{
  return "Gnuplot";
}

void Gnuplot::initialise(const std::string filename, const real interval,
    const unsigned int frequency)
{
  m_extension = ".dat";
  m_filename = filename;
  m_interval = interval;
  m_frequency = frequency;
  m_iterLast = 0;
  m_tLast = 0;
  m_frame = 0;
}

void Gnuplot::initialiseFromParameters(const Parameters& params)
{
  initialise(
    getParameter<std::string>(params,"filename"),
    getParameter<real>(params,"interval"),
    getParameter<unsigned int>(params,"frequency",std::numeric_limits<unsigned int>::max())
    );
}

void Gnuplot::output(const std::shared_ptr<DataPatch> data, const real t, 
            const unsigned int iter)
{
  // Consider putting this check in another function which is in Output.cpp which then calls this one
  if(m_iterLast == ((signed int) iter) && iter!=0)
  {
    return;
  }
  std::ostringstream frame;
  frame << std::setfill('0') << std::setw(5) << m_frame;
  std::ostringstream filename;
  filename << m_filename << "_" << frame.str() << m_extension;

  std::ofstream file(filename.str().c_str());

  if(file.is_open())
  {
    BOOST_LOG_TRIVIAL(info) << "Starting output " << filename.str();
    auto domain = ModuleList::uniqueModule<Domain>();

    file << "# version: " << GIT_ID << std::endl;
    file << "# t = " << t << std::endl;
    file << "# 1: x" << std::endl;
    file << "# 2: y" << std::endl;
    file << "# 3: n_x" << std::endl;
    file << "# 4: n_y" << std::endl;
    file << "# 5: s" << std::endl;

    unsigned int column = 6;
    const SystemAttributes::VariableNames names = ModuleList::uniqueModule<System>()->variableNames();
    for(unsigned int i=0;i<names.size();i++)
    {
      file << "# " << column << ": " << names[i] << std::endl;
      column++;
    }

    const DerivedVariablesMap derivedVariables = ModuleList::uniqueModule<System>()->derivedVariables();
    for( auto derivedVariablePair: derivedVariables)
    {
      file << "# " << column << ": " << derivedVariablePair.first << std::endl;
      column++;
    }

    for(unsigned int cell=0; cell<data->rows();cell++)
    {
      const real s = domain->s(cell);
      const Coord x = domain->x(s);
      const Coord n = domain->n(s);

      // Write positions
      file << std::setprecision(std::numeric_limits<real>::digits10)
           << x[0] << " " << x[1] << " " << n[0] << " " << n[1] << " " << s << " ";

      // Write state
      const Eigen::Map<State> state = Eigen::Map<State>(&((*data)(cell,0)));
      for(unsigned int i=0; i<SystemAttributes::stateSize;i++)
      {
        file << std::setprecision(std::numeric_limits<real>::digits10) << state[i] << " ";
      }

      // Write derived variables
      for( auto derivedVariablePair: derivedVariables)
      {
        auto derivedVar = derivedVariablePair.second;
        file << std::setprecision(std::numeric_limits<real>::digits10) << derivedVar(state,x,n);
      }

      file << std::endl;	
    }
    file.close();
    BOOST_LOG_TRIVIAL(info) << "Finished output " << filename.str();
  }
  else
  {
    BOOST_LOG_TRIVIAL(error) << "Cannot open file " << filename.str();
  }
  m_frame++;
  m_iterLast = iter;
  m_tLast = t;
}
