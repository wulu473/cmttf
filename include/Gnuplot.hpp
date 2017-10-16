
#ifndef GNUPLOT_H_
#define GNUPLOT_H_

#include "Output.hpp"

class Gnuplot : public Output
{
  REGISTER(Gnuplot);
  public:
    virtual std::string moduleName() const;
    Gnuplot();
    virtual ~Gnuplot();

    virtual void initialise(const std::string filename, const real interval, 
                            const unsigned int frequency);
    virtual void initialiseFromFile();


    virtual void output(std::shared_ptr<DataPatch> data, const real t, 
                        const unsigned int iter);
};

#endif
