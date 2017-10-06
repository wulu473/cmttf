
/*
 * Creator.cpp
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
#include "Creator.hpp"
#include "Factory.hpp"

Creator::Creator(const std::string& classname)
{
  Factory::registerit(classname, this);
}

