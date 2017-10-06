
#ifndef DATAPATCH_H_
#define DATAPATCH_H_

#include <Eigen/Dense>

#include "Attributes.hpp"

typedef Eigen::Matrix<real,Eigen::Dynamic,SystemAttributes::stateSize> DataPatchBase;

class DataPatch : public DataPatchBase
{
  public:
    //! Make vector constructor available
    DataPatch(const unsigned int size) : DataPatchBase(size,SystemAttributes::stateSize) {}
};

#endif
