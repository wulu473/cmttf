

#include <boost/test/unit_test.hpp>

#include "Expression.hpp"
#include "Attributes.hpp"
#include "ModuleList.hpp"
#include "Flat.hpp"
#include "System.hpp"

#ifndef ROBERTS
#warning Expression unit tests only work for system roberts
#warning Skipping ExpressionTest.cpp ...
#else

BOOST_AUTO_TEST_SUITE(ExpressionTests)

BOOST_AUTO_TEST_CASE(setData)
{
  std::shared_ptr<Flat> flat = std::make_shared<Flat>();
  flat->initialise(5,0.,0.5);
  ModuleList::addModule(flat);

  Expression::StateExpr expr; 
  expr[0] = TimeSpaceDependReal("x < 0.2 ? 0. : 1.");
  expr[1] = TimeSpaceDependReal("2.");

  Expression ic;
  ic.initialise(expr);

  std::shared_ptr<DataPatch> data = std::make_shared<DataPatch>(5);

  ic.setData(data);

  BOOST_CHECK_CLOSE((*data)(0,0),0,1e-5);
  BOOST_CHECK_CLOSE((*data)(0,1),2,1e-5);
  BOOST_CHECK_CLOSE((*data)(2,0),1,1e-5);
  BOOST_CHECK_CLOSE((*data)(2,1),2,1e-5);
  BOOST_CHECK_CLOSE((*data)(4,0),1,1e-5);
  BOOST_CHECK_CLOSE((*data)(4,1),2,1e-5);
}

BOOST_AUTO_TEST_SUITE_END()
#endif

