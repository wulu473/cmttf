/*
 * CreatorImpl.h
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
#ifndef _CREATORIMPL_H_
#define _CREATORIMPL_H_

#include <string>

#include "Creator.hpp"

template <class T>
class CreatorImpl : public Creator
{
  public:
    //! @todo We could pass here a function pointer to the name() or moduleName() method instead of letting the developer make sure moduleName() and REGISTERIPL match
    CreatorImpl<T>(const std::string& classname) : Creator(classname) {}
    virtual ~CreatorImpl<T>() {}

    virtual std::shared_ptr<ParameterisedModuleBase> create()
    { 
      return std::make_shared<T>();
    }
};

#endif //_CREATORIMPL_H_

