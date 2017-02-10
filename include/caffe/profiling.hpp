#pragma once
#include "caffe/export.hpp"
namespace caffe
{
    struct CAFFE_EXPORT scoped_profile
    {
        scoped_profile(const char* name);
        ~scoped_profile();
    };
}

#define PROFILE_LAYER scoped_profile profile_layer(layer_param().name().c_str());

