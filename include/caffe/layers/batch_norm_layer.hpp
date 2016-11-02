#ifndef CAFFE_BATCHNORM_LAYER_HPP_
#define CAFFE_BATCHNORM_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe_pb.h"

#define BN_VARIANCE_CLIP_START 200
#define BN_VARIANCE_CLIP_CONST 4.0

namespace caffe {

/**
 * @brief Normalizes the input to have 0-mean and/or unit (1) variance across
 *        the batch.
 *
 * This layer computes Batch Normalization as described in [1]. For each channel
 * in the data (i.e. axis 1), it subtracts the mean and divides by the variance,
 * where both statistics are computed across both spatial dimensions and across
 * the different examples in the batch.
 *
 * By default, during training time, the network is computing global
 * mean/variance statistics via a running average, which is then used at test
 * time to allow deterministic outputs for each input. You can manually toggle
 * whether the network is accumulating or using the statistics via the
 * use_global_stats option. For reference, these statistics are kept in the
 * layer's three blobs: (0) mean, (1) variance, and (2) moving average factor.
 *
 * Note that the original paper also included a per-channel learned bias and
 * scaling factor. To implement this in Caffe, define a `ScaleLayer` configured
 * with `bias_term: true` after each `BatchNormLayer` to handle both the bias
 * and scaling factor.
 *
 * [1] S. Ioffe and C. Szegedy, "Batch Normalization: Accelerating Deep Network
 *     Training by Reducing Internal Covariate Shift." arXiv preprint
 *     arXiv:1502.03167 (2015).
 *
 * TODO(dox): thorough documentation for Forward, Backward, and proto params.
 */
template <typename Dtype>
class CAFFE_EXPORT BatchNormLayer : public Layer<Dtype> {
 public:
  explicit BatchNormLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "BatchNorm"; }
  virtual inline int ExactNumBottomBlobs() const { return 1; }
  virtual inline int ExactNumTopBlobs() const { return 1; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
     const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  virtual void multicast_cpu(int N, int C, int S, const Dtype* x, Dtype* y);
  virtual void compute_sum_per_channel_cpu(int N, int C, int S,
      const Dtype* x, Dtype* y);
  virtual void compute_mean_per_channel_cpu(int N, int C, int S,
      const Dtype* x, Dtype* y);
#ifndef CPU_ONLY
  virtual void compute_sum_per_channel_gpu(int N, int C, int S,
      const Dtype *x, Dtype* y);
  virtual void multicast_gpu(int N, int C, int S, const Dtype* x, Dtype* y);
  virtual void compute_mean_per_channel_gpu(int N, int C, int S,
      const Dtype* x, Dtype* y);
#endif

  Blob<Dtype> mean_, variance_, inv_variance_, x_norm_;
  bool use_global_stats_;
  Dtype moving_average_fraction_;
  int channels_;
  Dtype eps_;
  int iter_;
  // auxiliary arrays
  Blob<Dtype> ones_N_, ones_HW_, ones_C_;
  Blob<Dtype> temp_;
  Blob<Dtype> temp_C_;
  Blob<Dtype> temp_NC_;
};

}  // namespace caffe

#endif  // CAFFE_BATCHNORM_LAYER_HPP_
