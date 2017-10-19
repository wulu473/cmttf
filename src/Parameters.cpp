#include "Parameters.hpp"

#include <sstream>

#include "TimeSpaceDependReal.hpp"

libconfig::Config Parameters::m_cfg;

void Parameters::readFile(const std::string& fileName)
{
  try
  {
    m_cfg.readFile(fileName.c_str());
  }
  catch( const libconfig::FileIOException &fioex)
  {
    BOOST_LOG_TRIVIAL(error) << "I/O error while reading file " << fileName;
    exit(10);
  }
  catch( const libconfig::ParseException &pex)
  {
    BOOST_LOG_TRIVIAL(error) << "Parse error at " << pex.getFile() << ":" << pex.getLine() << " - " << pex.getError();
    exit(10);
  }
}

bool Parameters::exists(const std::string& name)
{
  if(!m_cfg.exists(name))
  {
    return false;
  }
  std::ostringstream n;
  n << name << ".active";
  const bool isActive = getParameter<bool>(n.str(),true);
  return isActive;
}

int Parameters::getLength(const std::string& name)
{
  if(m_cfg.exists(name))
  {
    libconfig::Setting &stg = m_cfg.lookup(name);
    return stg.getLength();
  }
  else
  {
    return -1;
  }
}

std::string Parameters::activeModule(const std::string& parentModule)
{
  std::list<std::string> moduleNames = activeModules(parentModule);
  if (moduleNames.size() > 1)
  {
    BOOST_LOG_TRIVIAL(error) << "Multiple active modules for " << parentModule << "found.";
    exit(10);
  }
  if (moduleNames.empty())
  {
    return "";
  }
  return moduleNames.front();	
}

std::list<std::string> Parameters::activeModules(const std::string& parentModule)
{
  std::list<std::string> moduleNames;
  try
  {
    libconfig::Setting& settings = m_cfg.lookup(parentModule);
    for(int i=0 ;i<settings.getLength();i++)
    {
      if(settings[i]["active"])
      {
        moduleNames.push_back(settings[i].getName());
      } 
    }	
  }
  catch (const libconfig::SettingNotFoundException &stex)
  {
    BOOST_LOG_TRIVIAL(error) << "No active module for " << parentModule << "found.";
    exit(10);
  }
  return moduleNames;	
}

//! Return list of all active modules
/* Iterate through all modules in the settings file and return a list of active ones
 */
std::list<std::string> Parameters::allActiveModules()
{
  std::list<std::string> moduleNames;
  libconfig::Setting& root = m_cfg.getRoot();
  for(int i_rt=0; i_rt<root.getLength();i_rt++)
  {
    libconfig::Setting& base = root[i_rt];
    for(int i_base=0; i_base<base.getLength();i_base++)
    {
      libconfig::Setting& mod = base[i_base];  
      bool isModule=false;
      mod.lookupValue("active",isModule);
      if(isModule)
      { 
        moduleNames.push_back(mod.getName());
      }
    }
  }
  return moduleNames;
}


//! Read an expression parameter from settings file. If not supplied abort.
/** The parameter assigned to name has to be a string and can depend on time t and position x
 *
 * E.g. rho="1.225*exp(-x*x)"
 *
 * The return value is a std::function and can be evaluated by passing t and x (in that order)
 *
 */
const TimeSpaceDependReal
Parameters::getExpressionParameter(const std::string& name)
{
  const std::string param_str = getParameter<std::string>(name);
  return TimeSpaceDependReal(param_str);
}

