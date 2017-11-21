#ifndef PARAMETERS_H_
#error ParametersTemplates.cpp must be included in Parameters.H
#endif

#include <boost/log/trivial.hpp>
#include <iostream>
#include <sstream>
#include <typeinfo>

// Read parameter from settings file. If not supplied abort.
template<typename T>
const T Parameters::getParameter(const std::string& name) const
{
  T value;
  /*
   * If the following call fails with an error message something like
   *
   *./ThinFilm-test: symbol lookup error: ./ThinFilm-test: undefined symbol: _ZNK9libconfig6Config11lookupValueEPKcRNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE
   *
   * Check if the LD_LIBRARY_PATH is set correctly
   */
  if ( !m_cfg.lookupValue(name,value))
  {
    BOOST_LOG_TRIVIAL(error) << "Undefined parameter " << name;
    exit(1);
  }
  return value;
}

// Read parameter from settings file. If not supplied use default value
template<typename T>
const T Parameters::getParameter(const std::string& name, const T& defaultValue) const
{
  T value;
  if ( !m_cfg.lookupValue(name,value)) {
    BOOST_LOG_TRIVIAL(warning) << "Undefined parameter " << name << ": use default: " << defaultValue;
    value = defaultValue;
  }
  return value;
}

// Read parameter from settings file. If not supplied abort.
template<typename T>
const std::vector<T> Parameters::getVectorParameter(const std::string& name) const
{
  std::vector<T> values;

  if(!m_cfg.exists(name)) {
    BOOST_LOG_TRIVIAL(error) << "Undefined parameter " << name;
    exit(1);
  }

  const libconfig::Setting &stg = m_cfg.lookup(name);
  const int length = stg.getLength();

  for(int i=0;i<length;i++)
  {
    try {
      values.push_back(stg[i]);
    } catch (const libconfig::SettingTypeException &stex) {
      BOOST_LOG_TRIVIAL(error) << "Parameter type in " << name << " not matching. Parameter should be of type " << typeid(T).name();
      exit(1);
    }
  }
  std::ostringstream msg;
  msg << "Parameter: " << name << " Values: (";
  for(unsigned int i=0;i<values.size();i++)
  {
    msg << " " << values[i] << " ,";
  }
  msg << '\b' << ")";
  BOOST_LOG_TRIVIAL(debug) << msg.str();
  return values;
}


template<typename T>
void Parameters::deleteUnactive(std::list<T*>& list)
{
  typename std::list<T*>::iterator it = list.begin();
  while (it != list.end())
  {
    const std::string name = (*it)->name();
    BOOST_LOG_TRIVIAL(debug) << "Available module: " << name;
    if(!exists(name))
    {
      BOOST_LOG_TRIVIAL(debug) << "Deleting module " << name;
      delete (*it);
      it = list.erase(it); // erase also increments the iterator
    } 
    else 
    {
      ++it; // increment
    }
  }
}


