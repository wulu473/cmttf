#ifndef SYSTEMTESTUTILITY_H_
#define SYSTEMTESTUTILITY_H_

#include <memory>
#include "Attributes.hpp"

class System;

namespace SystemTestUtility
{
  bool checkJacobianNumerically(std::shared_ptr<System> sys, const StencilArray& u, const real dx,
      const real x, const real t, const real eps);
}

#endif
