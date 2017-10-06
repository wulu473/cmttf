/*
 * Creator.h
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
#ifndef _CREATOR_H_
#define _CREATOR_H_

#include <string>
#include <memory>

class ParameterisedModuleBase;

class Creator
{
public:
	Creator(const std::string& classname);
	virtual ~Creator() {};

	virtual std::shared_ptr<ParameterisedModuleBase> create() = 0;
};

#endif //_CREATOR_H_

