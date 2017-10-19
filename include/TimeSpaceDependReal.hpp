
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

class TimeSpaceDependReal : public std::function<real(real,real)>
{
  protected:
    typedef std::function<real(real,real)> TSDRBase;
  public:
    TimeSpaceDependReal() : TSDRBase() {}
    TimeSpaceDependReal(const TSDRBase& f) : TSDRBase(f) {}
    TimeSpaceDependReal(const std::string&);

  protected:
    static TimeSpaceDependReal createFromString(const std::string&);
};

#endif
