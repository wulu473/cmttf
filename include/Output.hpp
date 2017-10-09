
#ifndef OUTPUT_H_
#define OUTPUT_H_

#include "ModuleBase.hpp"
#include "Attributes.hpp"
#include "DataPatch.hpp"

class Output : public ParameterisedModuleBase
{
  public:
    virtual std::string baseName() const final;

    Output();
    virtual ~Output();

    virtual void output(std::shared_ptr<DataPatch> data, const real t, 
                        const unsigned int iter) = 0;

    virtual bool needsOutput(const real t, const real dt, const unsigned int iter) const;

    virtual real maxDt(const real t, const real dt, const unsigned int iter) const;
  protected:
    int m_iterLast;
    real m_tLast;
    unsigned int m_frame;

    real m_interval;
    unsigned int m_frequency;

    std::string m_filename;
    std::string m_extension;
};

#endif
