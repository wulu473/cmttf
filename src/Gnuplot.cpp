
#include <fstream>
#include <sstream>
#include <iomanip>

#include "ModuleList.hpp"
#include "Domain.hpp"
#include "Git.hpp"

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

void Gnuplot::initialiseFromFile()
{
  initialise(
    getParameter<std::string>("filename"),
    getParameter<real>("interval"),
    getParameter<unsigned int>("frequency",std::numeric_limits<unsigned int>::max())
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
    /*
     * TODO Print variable names
    const std::vector<std::string> names = ModuleList::uniqueModule<System>()->variableNames();
    for(unsigned int i=0;i<names.size();i++)
      file << "# " << i+2 << ": " << names[i] << std::endl;
    */

    for(unsigned int cell=0; cell<data->rows();cell++)
    {
      const real s = domain->s(cell);
      const Coord x = domain->x(s);

      file << std::setprecision(std::numeric_limits<real>::digits10) << x[0] << " " << x[1] << " ";
      for(unsigned int i=0; i<SystemAttributes::stateSize;i++)
      {
        file << std::setprecision(std::numeric_limits<real>::digits10) << (*data)(cell,i) << " ";
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
