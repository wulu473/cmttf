
#include "TimeSpaceDependReal.hpp"

#include "Log.hpp"

#include <memory>

TimeSpaceDependReal::TimeSpaceDependReal(const std::string& exp_str)
{
  *this = createFromString(exp_str);
}

TimeSpaceDependReal TimeSpaceDependReal::createFromString(const std::string& exp_str)
{
  std::shared_ptr<TimeSpaceDependRealWrapper> tsdr = std::make_shared<TimeSpaceDependRealWrapper>();

  tsdr->symbolTable.add_variable("s",tsdr->s);
  tsdr->symbolTable.add_variable("t",tsdr->t);

  tsdr->expression.register_symbol_table(tsdr->symbolTable);

  if(!tsdr->parser.compile(exp_str,tsdr->expression))
  {
    BOOST_LOG_TRIVIAL(error) << "Parsing error in expression: " << exp_str;

    // More detailed diagnostics
    for (std::size_t i = 0; i < tsdr->parser.error_count(); ++i)
    {
      exprtk::parser_error::type error = tsdr->parser.get_error(i);
      BOOST_LOG_TRIVIAL(error) << "Error[" << i << "] Position: " << error.token.position 
                               << " Type: [" << exprtk::parser_error::to_str(error.mode).c_str()
                               << "] Msg: " << error.diagnostic.c_str();
    }

    exit(1);
  }

  TSDRBase expression = [tsdr](real t, real s) 
  {
    tsdr->s = s;
    tsdr->t = t; return tsdr->expression.value();
  };

  return TimeSpaceDependReal(expression);
}
