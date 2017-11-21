/*
 * Factory.h
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
#ifndef FACTORY_H_
#define FACTORY_H_

#include <string>
#include <map>

#include "CreatorImpl.hpp"
#include "ModuleBase.hpp"

class Creator;

class Factory
{
  public:
    template<typename T>
    static std::shared_ptr<T> create(const std::string& classname);

    template<typename T>
    static std::shared_ptr<T> createInitialised(const std::string& classname, const Parameters& params);

    static void registerit(const std::string& classname, Creator* creator);
  private:
    static std::map<std::string, Creator*>& get_table();

    static std::shared_ptr<ParameterisedModuleBase> createBase(const std::string& classname);
};

#define REGISTER(classname) \
	private: \
	static const CreatorImpl<classname> creator;

#define REGISTERIMPL(classname) \
	const CreatorImpl<classname> classname::creator(#classname);

#define REGISTERIMPL_DIFF_NAME(classname,name) \
	const CreatorImpl<classname> classname::creator(#name);

#include "FactoryTemplates.cpp"
#endif //_FACTORY_H_

