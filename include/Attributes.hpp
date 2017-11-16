
#ifndef ATTRIBUTES
#define ATTRIBUTES

#include <Eigen/Dense>
#include <string>

#include "SystemAttributes.hpp"

typedef double real;

using Eigen::Ref;
typedef Eigen::Matrix<real,SystemAttributes::stateSize,1> State;
typedef Eigen::Array<State,SystemAttributes::stencilSize*2+1,1> StencilArray;
typedef Eigen::Array<real,SystemAttributes::stateSize,SystemAttributes::stateSize*(SystemAttributes::stencilSize*2+1)> StencilJacobian;
typedef Eigen::Array<State,Eigen::Dynamic,1> StateArray;

typedef Eigen::Matrix<real,2,1> Coord;

class NotImplemented : public std::exception
{

};

#endif

