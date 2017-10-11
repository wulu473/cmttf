#ifndef TESTUTILITY_H_
#define TESTUTILITY_H_

#include <memory>
#include "Attributes.hpp"

class System;

namespace TestUtility
{
  namespace SystemTest
  {
    bool checkJacobianNumerically(std::shared_ptr<System> sys, const StencilArray& u, const real dx,
        const real x, const real t, const real eps);
  }
}

#endif
