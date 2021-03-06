#ifndef SYSTEMATTRIBUTES_H_
#define SYSTEMATTRIBUTES_H_

namespace SystemAttributes
{
    const unsigned int stateSize = 2;
    const unsigned int stencilSize = 2;

    typedef std::array<std::string, stencilSize> VariableNames;
};

#endif

