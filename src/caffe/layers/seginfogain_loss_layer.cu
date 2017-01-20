#include "caffe/layers/seginfogain_loss_layer.hpp"
#include "cuda.h"
#include "vector_types.h"
#include "cuda_runtime.h"

#define CUDA_KERNEL_LOOP_N(i, n, Dim) \
  for (int i = blockIdx.Dim * blockDim.Dim + threadIdx.Dim; \
       i < (n); \
       i += blockDim.Dim * gridDim.Dim)

template<typename Dtype>
void __global__  seginfo_gain_loss_forward_kernel(const Dtype* label, const int max_label,
                                         const int ignore_label, const int numLabels,
                                         const int outer_num_, const int inner_num_,
                                         const Dtype* infogain_mat, Dtype* loss, Dtype* count,
                                                  const Dtype* prob_data, const int dim)
{
    CUDA_KERNEL_LOOP(i, outer_num_)
    {
        for(int j = 0; j < inner_num_; ++j)
        {
            const int label_value = static_cast<int>(label[i * inner_num_ + j]);
            if(label_value == ignore_label ||
                label_value < 0 || label_value >= max_label)
                continue;
            for (int k = 0; k < numLabels; k++)
            {
                atomicAdd(loss,- (infogain_mat[label_value * numLabels + k] *
                        log(max(prob_data[i * dim + k * inner_num_ + j],
                            Dtype(caffe::kLOG_THRESHOLD)))));
            }
            atomicAdd(count, (Dtype)1.0);
        }
    }
}

template<typename Dtype>
void __global__ seginfo_gain_loss_backwards_kernel(const Dtype* label, const int numLabels,
                                                   const int ignore_label_, Dtype* bottom_diff,
                                                   const Dtype* infogain_mat,
                                                   const Dtype* infogain_sum, Dtype* count,
                                                   const int channels,
                                                   const int outer_num_, const int inner_num_,
                                                   const int dim)
{

    CUDA_KERNEL_LOOP(i, outer_num_) {
        for (int j = 0; j < inner_num_; ++j) {
            const int label_value = static_cast<int>(label[i * inner_num_ + j]);
            if (label_value == ignore_label_) {
                for (int c = 0; c < channels; ++c) {
                    bottom_diff[i * dim + c * inner_num_ + j] = 0;
                }
            }
            else {

                for (int k = 0; k < numLabels; k++)
                {
                    bottom_diff[i * dim + k * inner_num_ + j] *= infogain_sum[label_value];
                    bottom_diff[i * dim + k * inner_num_ + j] -= infogain_mat[label_value * numLabels + k];
                }
                //++count;
                atomicAdd(count, (Dtype)1.0);
            }
        }
    }
}

namespace caffe{
  template<typename Dtype>
  void SegInfogainLossLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top){
    softmax_layer_->Forward(bottom, top);
    const Dtype* prob_data = prob_.gpu_data();
    const Dtype* label = bottom[1]->gpu_data();

    const Dtype* infogain_mat = NULL;
    if (bottom.size() < 3) {
        infogain_mat = infogain_.gpu_data();
    }
    else {
        infogain_mat = bottom[2]->gpu_data();
    }
    int dim = prob_.count() / outer_num_; //step between cases: number of voxels*labels
    int numLabels = prob_.count() / outer_num_ / inner_num_;

    Dtype* loss = 0;
    Dtype* count = 0;
    cudaMalloc(&loss, sizeof(Dtype));
    cudaMemset(loss, 0, sizeof(Dtype));
    cudaMalloc(&count, sizeof(Dtype));
    cudaMemset(count, 0, sizeof(Dtype));
    seginfo_gain_loss_forward_kernel<Dtype><<<CAFFE_GET_BLOCKS(outer_num_ * inner_num_),
            CAFFE_CUDA_NUM_THREADS>>>(label, prob_.shape(softmax_axis_),
                                               has_ignore_label_? ignore_label_ : -1,
                                               numLabels, outer_num_, inner_num_,
                                               infogain_mat, loss, count, prob_data, dim);
    cudaFree(loss);
    cudaFree(count);

    if (normalize_) {
        //top[0]->mutable_cpu_data()[0] = loss / count;
        caffe_gpu_div(1, loss, count, top[0]->mutable_gpu_data());
    }
    else {
        //top[0]->mutable_cpu_data()[0] = loss / outer_num_;
        caffe_gpu_scale(1, 1.0f / (Dtype)outer_num_, loss, top[0]->mutable_gpu_data());
    }
    if (top.size() == 2) {
        top[1]->ShareData(prob_);
    }
  }

  template<typename Dtype>
  void SegInfogainLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
      if (propagate_down[1]) {
          LOG(FATAL) << this->type()
              << " Layer cannot backpropagate to label inputs.";
      }
      if (propagate_down.size() > 2 && propagate_down[2]) {
          LOG(FATAL) << this->type()
              << " Layer cannot backpropagate to infogain inputs.";
      }
      if (propagate_down[0]) {
          Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
          const Dtype* prob_data = prob_.gpu_data();

          //caffe_gpu_copy(prob_.count(), prob_data, bottom_diff);
          cudaMemcpy(bottom_diff, prob_data, prob_.count() * sizeof(Dtype), cudaMemcpyDeviceToDevice);


          const Dtype* label = bottom[1]->gpu_data();
          const Dtype* infogain_mat = NULL;
          Dtype* infogain_sum = infogain_sum_.mutable_gpu_data();

          int numLabels = prob_.count() / outer_num_ / inner_num_;
          int dim = prob_.count() / outer_num_;
          if (bottom.size() < 3) {
              infogain_mat = infogain_.gpu_data();
          }
          else {
              infogain_mat = bottom[2]->gpu_data();
              for (int labelIt = 0; labelIt < numLabels; labelIt++)
              {
                  //infogain_sum[labelIt] = caffe_gpu_asum(numLabels, infogain_mat + labelIt * numLabels);
                  caffe_gpu_asum(numLabels, infogain_mat + labelIt * numLabels, infogain_sum + labelIt);
              }
          }
          const Dtype* infogainSum = infogain_sum_.gpu_data();
          Dtype* count = 0;
          cudaMalloc(&count, sizeof(Dtype));
          cudaMemset(count, 0, sizeof(Dtype));
          seginfo_gain_loss_backwards_kernel<<<CAFFE_GET_BLOCKS(outer_num_ * inner_num_),
                  CAFFE_CUDA_NUM_THREADS>>>(label, numLabels, has_ignore_label_ ? ignore_label_ : -1,
                                             bottom_diff, infogain_mat, infogainSum, count, bottom[0]->shape(softmax_axis_),
                                            outer_num_, inner_num_, dim);

          // Scale gradient
          const Dtype loss_weight = top[0]->cpu_diff()[0];
          if (normalize_) {
              //caffe_scal(prob_.count(), loss_weight / count, bottom_diff);
              Dtype h_count;
              cudaMemcpy(&h_count, count, sizeof(Dtype), cudaMemcpyDeviceToHost);
              caffe_gpu_scal(prob_.count(), loss_weight / h_count, bottom_diff);

          }
          else {
              //caffe_scal(prob_.count(), loss_weight / outer_num_, bottom_diff);
              caffe_gpu_scal(prob_.count(), loss_weight / outer_num_, bottom_diff);
          }
      }
  }
  INSTANTIATE_LAYER_GPU_FUNCS(SegInfogainLossLayer);
}



