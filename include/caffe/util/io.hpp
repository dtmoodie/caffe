#ifndef CAFFE_UTIL_IO_H_
#define CAFFE_UTIL_IO_H_

#include <boost/filesystem.hpp>
#include <iomanip>
#include <iostream>  // NOLINT(readability/streams)
#include <string>

#include "google/protobuf/message.h"

#include "caffe/common.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/format.hpp"


#ifndef CAFFE_TMP_DIR_RETRIES
#define CAFFE_TMP_DIR_RETRIES 100
#endif

namespace caffe {

using ::google::protobuf::Message;
using ::boost::filesystem::path;

inline void MakeTempDir(string* temp_dirname) {
  temp_dirname->clear();
  const path& model =
    boost::filesystem::temp_directory_path()/"caffe_test.%%%%-%%%%";
  for ( int i = 0; i < CAFFE_TMP_DIR_RETRIES; i++ ) {
    const path& dir = boost::filesystem::unique_path(model).string();
    bool done = boost::filesystem::create_directory(dir);
    if ( done ) {
      *temp_dirname = dir.string();
      return;
    }
  }
  LOG(FATAL) << "Failed to create a temporary directory.";
}

inline void MakeTempFilename(string* temp_filename) {
  static path temp_files_subpath;
  static uint64_t next_temp_file = 0;
  temp_filename->clear();
  if ( temp_files_subpath.empty() ) {
    string path_string="";
    MakeTempDir(&path_string);
    temp_files_subpath = path_string;
  }
  *temp_filename =
    (temp_files_subpath/caffe::format_int(next_temp_file++, 9)).string();
}



bool DLL_EXPORT ReadProtoFromTextFile(const char* filename, Message* proto);

inline bool DLL_EXPORT ReadProtoFromTextFile(const string& filename, Message* proto) {
  return ReadProtoFromTextFile(filename.c_str(), proto);
}

inline void DLL_EXPORT ReadProtoFromTextFileOrDie(const char* filename, Message* proto) {
  CHECK(ReadProtoFromTextFile(filename, proto));
}

inline void DLL_EXPORT ReadProtoFromTextFileOrDie(const string& filename, Message* proto) {
  ReadProtoFromTextFileOrDie(filename.c_str(), proto);
}

void DLL_EXPORT WriteProtoToTextFile(const Message& proto, const char* filename);
inline void DLL_EXPORT WriteProtoToTextFile(const Message& proto, const string& filename) {
  WriteProtoToTextFile(proto, filename.c_str());
}

bool DLL_EXPORT ReadProtoFromBinaryFile(const char* filename, Message* proto);

inline bool DLL_EXPORT ReadProtoFromBinaryFile(const string& filename, Message* proto) {
  return ReadProtoFromBinaryFile(filename.c_str(), proto);
}

inline void DLL_EXPORT ReadProtoFromBinaryFileOrDie(const char* filename, Message* proto) {
  CHECK(ReadProtoFromBinaryFile(filename, proto));
}

inline void DLL_EXPORT ReadProtoFromBinaryFileOrDie(const string& filename,
                                         Message* proto) {
  ReadProtoFromBinaryFileOrDie(filename.c_str(), proto);
}


void DLL_EXPORT WriteProtoToBinaryFile(const Message& proto, const char* filename);
inline void DLL_EXPORT WriteProtoToBinaryFile(
    const Message& proto, const string& filename) {
  WriteProtoToBinaryFile(proto, filename.c_str());
}

bool DLL_EXPORT ReadFileToDatum(const string& filename, const int label, Datum* datum);

inline bool DLL_EXPORT ReadFileToDatum(const string& filename, Datum* datum) {
  return ReadFileToDatum(filename, -1, datum);
}

bool DLL_EXPORT ReadImageToDatum(const string& filename, const int label,
    const int height, const int width, const bool is_color,
    const std::string & encoding, Datum* datum);

inline bool DLL_EXPORT ReadImageToDatum(const string& filename, const int label,
    const int height, const int width, const bool is_color, Datum* datum) {
  return ReadImageToDatum(filename, label, height, width, is_color,
                          "", datum);
}

inline bool DLL_EXPORT ReadImageToDatum(const string& filename, const int label,
    const int height, const int width, Datum* datum) {
  return ReadImageToDatum(filename, label, height, width, true, datum);
}

inline bool DLL_EXPORT ReadImageToDatum(const string& filename, const int label,
    const bool is_color, Datum* datum) {
  return ReadImageToDatum(filename, label, 0, 0, is_color, datum);
}

inline bool DLL_EXPORT ReadImageToDatum(const string& filename, const int label,
    Datum* datum) {
  return ReadImageToDatum(filename, label, 0, 0, true, datum);
}

inline bool DLL_EXPORT ReadImageToDatum(const string& filename, const int label,
    const std::string & encoding, Datum* datum) {
  return ReadImageToDatum(filename, label, 0, 0, true, encoding, datum);
}

bool DLL_EXPORT DecodeDatumNative(Datum* datum);
bool DLL_EXPORT DecodeDatum(Datum* datum, bool is_color);

#ifdef USE_OPENCV
cv::Mat DLL_EXPORT ReadImageToCVMat(const string& filename,
    const int height, const int width, const bool is_color);

cv::Mat DLL_EXPORT ReadImageToCVMat(const string& filename,
    const int height, const int width);

cv::Mat DLL_EXPORT ReadImageToCVMat(const string& filename,
    const bool is_color);

cv::Mat DLL_EXPORT ReadImageToCVMat(const string& filename);

cv::Mat DLL_EXPORT DecodeDatumToCVMatNative(const Datum& datum);
cv::Mat DLL_EXPORT DecodeDatumToCVMat(const Datum& datum, bool is_color);

void DLL_EXPORT CVMatToDatum(const cv::Mat& cv_img, Datum* datum);
#endif  // USE_OPENCV
/*
/*template <typename Dtype>
void DLL_EXPORT hdf5_load_nd_dataset_helper(
    hid_t file_id, const char* dataset_name_, int min_dim, int max_dim,
    Blob<Dtype>* blob);

template <typename Dtype>
void DLL_EXPORT hdf5_load_nd_dataset(
    hid_t file_id, const char* dataset_name_, int min_dim, int max_dim,
    Blob<Dtype>* blob);

template <typename Dtype>
void DLL_EXPORT hdf5_save_nd_dataset(
    const hid_t file_id, const string& dataset_name, const Blob<Dtype>& blob);
	*/
}  // namespace caffe

#endif   // CAFFE_UTIL_IO_H_
