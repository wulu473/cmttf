/*
 * Factory.cpp
 * 
 *  Created on: April 14, 2013
 *      Author: David J. Rager
 *       Email: djrager@fourthwoods.com
 *
 * This code is hereby released into the public domain per the Creative Commons
 * Public Domain dedication.
 *
 * http://http://creativecommons.org/publicdomain/zero/1.0/
 */
#include "Factory.hpp"
#include "Creator.hpp"

void Factory::registerit(const std::string& classname, Creator* creator)
{
  get_table()[classname] = creator;
}
std::map<std::string, Creator*>& Factory::get_table()
{
  static std::map<std::string, Creator*> table;
  return table;
}


std::shared_ptr<ParameterisedModuleBase>
Factory::createBase(const std::string& classname)
{
  std::map<std::string, Creator*>::iterator i;
  i = get_table().find(classname);

  // Check if class is registered
  if(i == get_table().end())
  {
    BOOST_LOG_TRIVIAL(error) << "Trying to create not registered class. Class name: " << classname;
    exit(2);
  }

  return i->second->create();
}

