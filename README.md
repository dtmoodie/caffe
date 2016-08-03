# Caffe

This is a fork of caffe designed to support embedding in applications on windows.
Changes found in this fork:

Windows support through cmake builds.
Export declarations to allow shared library building.
Checks throw exceptions instead of aborting your application.
Boost log is used in place of glog.
Boost program_options used in place of gflags.
