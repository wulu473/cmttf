
#ifndef ATTRIBUTES
#define ATTRIBUTES

#include <Eigen/Dense>
#include <string>
#include <map>

#include "SystemAttributes.hpp"

typedef double real;

using Eigen::Ref;
typedef Eigen::Matrix<real,SystemAttributes::stateSize,1> State;
typedef Eigen::Array<State,SystemAttributes::stencilSize*2+1,1> StencilArray;
typedef Eigen::Array<real,SystemAttributes::stateSize,SystemAttributes::stateSize*(SystemAttributes::stencilSize*2+1)> StencilJacobian;
typedef Eigen::Array<State,Eigen::Dynamic,1> StateArray;

typedef Eigen::Matrix<real,2,1> Coord;


/*
 * Map that holds derived variables. The string is the name of the variable and the function is the way to compute it.
 * The function expects a state, x coordinate and normal of the cell
 *
 */
typedef std::pair<std::string, std::function<real(State,Coord,Coord)> > DerivedVariablePair;
typedef std::map<std::string, std::function<real(State,Coord,Coord)> > DerivedVariablesMap;

class NotImplemented : public std::exception
{

};

#endif

