#include "caffe/profiling.hpp"
#include "nvToolsExt.h"

using namespace caffe;


scoped_profile::scoped_profile(const char *name)
{
    nvtxRangePushA(name);
}
scoped_profile::~scoped_profile()
{
    nvtxRangePop();
}
