# Caffe

This is a fork of caffe designed to support embedding in applications on windows.
Changes found in this fork:

1) Windows support through cmake builds.

2) Export declarations to allow shared library building.

3) Checks throw exceptions instead of aborting your application.

4) Boost log is used in place of glog.

5) Boost program_options used in place of gflags.
