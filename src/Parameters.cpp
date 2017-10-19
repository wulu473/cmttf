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

//! Read an expression vector parameter from settings file. If not supplied abort.
/** The parameter assigned to name has to be a vector of string and can depend on time t and position x
 *
 * E.g. state = [ "1.225*exp(-x*x)" , "0." ]
 *
 * The return value is a std::vector of std::function and can be evaluated by passing t and x (in that order)
 *
 */
const std::vector<std::function<real(real,real)> >
Parameters::getExpressionVectorParameter(const std::string& name)
{
  std::vector<std::string> exp_str_vec = getVectorParameter<std::string>(name);

  std::vector<std::function<real(real,real)> > expression_vec;
  for(auto exp_str : exp_str_vec)
  {
    std::shared_ptr<TimeSpaceDependRealWrapper> tsdr = std::make_shared<TimeSpaceDependRealWrapper>();

    tsdr->symbolTable.add_variable("x",tsdr->x);
    tsdr->symbolTable.add_variable("t",tsdr->t);

    tsdr->expression.register_symbol_table(tsdr->symbolTable);

    if(!tsdr->parser.compile(exp_str,tsdr->expression))
    {
      BOOST_LOG_TRIVIAL(error) << "Parsing error in expression: " << exp_str;

      // More detailed diagnostics
      for (std::size_t i = 0; i < tsdr->parser.error_count(); ++i)
      {
        exprtk::parser_error::type error = tsdr->parser.get_error(i);
        BOOST_LOG_TRIVIAL(error) << "Error[" << i << "] Position: " << error.token.position 
                                 << " Type: [" << exprtk::parser_error::to_str(error.mode).c_str()
                                 << "] Msg: " << error.diagnostic.c_str();
      }

      exit(1);
    }

    std::function<real(real,real)> expression = [tsdr](real t, real x) 
    {
      tsdr->x = x;
      tsdr->t = t;
      return tsdr->expression.value();
    };
    expression_vec.push_back(expression);
  }

  return expression_vec;
}

