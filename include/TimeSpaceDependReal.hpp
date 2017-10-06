
#ifndef TIMESPACEDEPENDREAL_H_
#define TIMESPACEDEPENDREAL_H_

#include <exprtk.hpp>

#include "Attributes.hpp"

struct TimeSpaceDependRealWrapper
{
  real t, x;

  exprtk::symbol_table<real> symbolTable;
  exprtk::expression<real> expression;
  exprtk::parser<real> parser;
};

#endif
