#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layers/softmax_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void SoftmaxWithLossLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::LayerSetUp(bottom, top);
  LayerParameter softmax_param(this->layer_param_);
  softmax_param.set_type("Softmax");
  softmax_layer_ = LayerRegistry<Dtype>::CreateLayer(softmax_param);
  softmax_bottom_vec_.clear();
  softmax_bottom_vec_.push_back(bottom[0]);
  softmax_top_vec_.clear();
  softmax_top_vec_.push_back(&prob_);
  softmax_layer_->SetUp(softmax_bottom_vec_, softmax_top_vec_);

  has_ignore_label_ =
    this->layer_param_.loss_param().has_ignore_label();
  if (has_ignore_label_) {
    ignore_label_ = this->layer_param_.loss_param().ignore_label();
  }
  normalize_ = this->layer_param_.loss_param().normalize();
  if (!this->layer_param_.loss_param().has_normalization() &&
      this->layer_param_.loss_param().has_normalize()) {
    normalization_ = this->layer_param_.loss_param().normalize() ?
                     LossParameter_NormalizationMode_VALID :
                     LossParameter_NormalizationMode_BATCH_SIZE;
  } else {
    normalization_ = this->layer_param_.loss_param().normalization();
  }
  // weighted loss by Olaf and Mohamed Ezz
  class_loss_weights_.resize(this->layer_param_.loss_param().class_loss_weights_size());
  weigh_prediction_class_ = this->layer_param_.loss_param().weigh_prediction_class();

  if (bottom.size() == 3)
    DCHECK( !weigh_prediction_class_ )
      << "weigh_prediction_class is applicable only for class-wise weights."
      << " But a third blob was given to for pixel-wise weighting.";
  if (class_loss_weights_.size() > 2)
      DCHECK( !weigh_prediction_class_ )
        << "weigh_prediction_class is applicable only for binary class problems.";

  for( int i=0; i < class_loss_weights_.size(); ++i) {
    class_loss_weights_[i] = this->layer_param_.loss_param().class_loss_weights(i);
    std::cout << "class_loss_weights[" << i << "] = " << class_loss_weights_[i] << std::endl;
  }
}

template <typename Dtype>
void SoftmaxWithLossLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  softmax_layer_->Reshape(softmax_bottom_vec_, softmax_top_vec_);
  softmax_axis_ =
      bottom[0]->CanonicalAxisIndex(this->layer_param_.softmax_param().axis());

  outer_num_ = bottom[0]->count(0, softmax_axis_); // N
  inner_num_ = bottom[0]->count(softmax_axis_ + 1); // H * W
  CHECK_EQ(outer_num_ * inner_num_, bottom[1]->count())
      << "Number of labels must match number of predictions; "
      << "e.g., if softmax axis == 1 and prediction shape is (N, C, H, W), "
      << "label count (number of labels) must be N*H*W, "
      << "with integer values in {0, 1, ..., C-1}.";
  if (top.size() >= 2) {
    // softmax output
    top[1]->ReshapeLike(*bottom[0]);
  }
}


template <typename Dtype>
void SoftmaxWithLossLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  // The forward pass computes the softmax prob values.
  softmax_layer_->Forward(softmax_bottom_vec_, softmax_top_vec_);
  const Dtype* prob_data = prob_.cpu_data();
  const Dtype* label = bottom[1]->cpu_data();
  int num = prob_.num();
  int dim = prob_.count() / outer_num_; // n_classes * width * height
  Dtype weightsum = 0; // weights of valid points over which loss is calculated (valid = not in ignore_label)
  Dtype loss = 0;
  // Olaf Code Begins
  if( bottom.size() == 3) {
    // weighted version using a third input blob (by Olaf)
    const Dtype* weight_data = bottom[2]->cpu_data();
    for (int i = 0; i < num; ++i) {
      for (int j = 0; j < inner_num_; j++) {
        const int label_value = static_cast<int>(label[i * inner_num_ + j]);
        if (has_ignore_label_ && label_value == ignore_label_) {
          continue;
        }
        DCHECK_GE(label_value, 0);
        DCHECK_LT(label_value, prob_.channels());
        loss -= weight_data[j] *
            log(std::max(prob_data[i * dim + label_value * inner_num_ + j],
                         Dtype(FLT_MIN)));
        weightsum += weight_data[j];
      }
    }

    top[0]->mutable_cpu_data()[0] = loss / LossLayer<Dtype>::GetNormalizer(
                normalization_, outer_num_, inner_num_, weightsum);

  } else if(class_loss_weights_.size() > 0) {
    // weighted version using class-wise loss weights (by Olaf)
    for (int i = 0; i < num; ++i) {
      for (int j = 0; j < inner_num_; j++) {
        const int label_value = static_cast<int>(label[i * inner_num_ + j]);
        if (has_ignore_label_ && label_value == ignore_label_) {
          continue;
        }
        DCHECK_GE(label_value, 0);
        DCHECK_LT(label_value, prob_.channels());
        Dtype weight = 1;
        if( label_value < class_loss_weights_.size()) {
          weight = class_loss_weights_[label_value];
        }

        if (weigh_prediction_class_)
        {
          // Get predicted label
          Dtype max_prob = -1; //maximum probabaility found so far
          int pred = -1; // argmax probabilities (predicted class)
          for(int c = 0; c < 2; ++c) //loop over classes
          {
            Dtype current_prob = prob_data[i * dim + c * inner_num_ + j];
            if (current_prob > max_prob)
            {
              pred = c;
              max_prob = current_prob;
            }
        }
        if (pred == 1) // take weight of positive class
          weight = class_loss_weights_[1];
        }

        loss -= weight *
            log(std::max(prob_data[i * dim + label_value * inner_num_ + j],
                         Dtype(FLT_MIN)));
        weightsum += weight;
      }
    }
    top[0]->mutable_cpu_data()[0] = loss / LossLayer<Dtype>::GetNormalizer(
                normalization_, outer_num_, inner_num_, weightsum);
  } else {
    for (int i = 0; i < outer_num_; ++i) {
      for (int j = 0; j < inner_num_; j++) {
        const int label_value = static_cast<int>(label[i * inner_num_ + j]);
        if (has_ignore_label_ && label_value == ignore_label_) {
          continue;
        }
        DCHECK_GE(label_value, 0);
        DCHECK_LT(label_value, prob_.shape(softmax_axis_));
        loss -= log(std::max(prob_data[i * dim + label_value * inner_num_ + j],
                             Dtype(FLT_MIN)));
        ++weightsum;
      }
    }
  }
//<<<<<<< HEAD
  Dtype normalizer = LossLayer<Dtype>::GetNormalizer(
      normalization_, outer_num_, inner_num_, weightsum);
  top[0]->mutable_cpu_data()[0] = loss / normalizer;
/*=======
  top[0]->mutable_cpu_data()[0] = loss / get_normalizer(normalization_, weightsum);
>>>>>>> 83564bdf90c01da29daf42dfcc2bf7bce2f8a7fa*/
  if (top.size() == 2) {
    top[1]->ShareData(prob_);
  }
}

template <typename Dtype>
void SoftmaxWithLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[1]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to label inputs.";
  }
  if (propagate_down[0]) {
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    const Dtype* prob_data = prob_.cpu_data();
    caffe_copy(prob_.count(), prob_data, bottom_diff);
    const Dtype* label = bottom[1]->cpu_data();
    int dim = prob_.count() / outer_num_;
    int num = prob_.num();
    Dtype weightsum = 0; // sum of weights to normalize with
    // Olaf Code Begins
    if( bottom.size() == 3) {
       // weighted version using a third input blob (by Olaf)
      const Dtype* weight_data = bottom[2]->cpu_data();
      for (int i = 0; i < num; ++i) {
        for (int j = 0; j < inner_num_; ++j) {
          const int label_value = static_cast<int>(label[i * inner_num_ + j]);
          if (has_ignore_label_ && label_value == ignore_label_) {
            for (int c = 0; c < bottom[0]->channels(); ++c) {
              bottom_diff[i * dim + c * inner_num_ + j] = 0;
            }
          } else {
            bottom_diff[i * dim + label_value * inner_num_ + j] -= 1;
            for (int c = 0; c < bottom[0]->channels(); ++c) {
              bottom_diff[i * dim + c * inner_num_ + j] *= weight_data[j];
            }
            weightsum += weight_data[j];
          }
        }
      }
    } else if(class_loss_weights_.size() > 0) {
      // weighted version using class-wise loss weights (by Olaf)
      for (int i = 0; i < num; ++i) {
        for (int j = 0; j < inner_num_; ++j) {
          const int label_value = static_cast<int>(label[i * inner_num_ + j]);
          if (has_ignore_label_ && label_value == ignore_label_) {
            for (int c = 0; c < bottom[0]->channels(); ++c) {
              bottom_diff[i * dim + c * inner_num_ + j] = 0;
            }
          } else {
            Dtype weight = 1;
            if( label_value < class_loss_weights_.size()) {
              weight = class_loss_weights_[label_value];
            }

            if (weigh_prediction_class_)
            {
              // Get predicted label
              Dtype max_prob = -1; //maximum probabaility found so far
              int pred = -1; // argmax probabilities (predicted class)
              for(int c = 0; c < 2; ++c) //loop over classes
              {
                Dtype current_prob = prob_data[i * dim + c * inner_num_ + j];
                if (current_prob > max_prob)
                {
                  pred = c;
                  max_prob = current_prob;
                }
            }
            if (pred == 1) // take weight of positive class
              weight = class_loss_weights_[1];
            }

            bottom_diff[i * dim + label_value * inner_num_ + j] -= 1;

            for (int c = 0; c < bottom[0]->channels(); ++c) {
              bottom_diff[i * dim + c * inner_num_ + j] *= weight;
            }
           weightsum += weight;
          }
        }
      }

    } else {
      for (int i = 0; i < outer_num_; ++i) {
        for (int j = 0; j < inner_num_; ++j) {
          const int label_value = static_cast<int>(label[i * inner_num_ + j]);
          if (has_ignore_label_ && label_value == ignore_label_) {
            for (int c = 0; c < bottom[0]->shape(softmax_axis_); ++c) {
              bottom_diff[i * dim + c * inner_num_ + j] = 0;
            }
          } else {
            bottom_diff[i * dim + label_value * inner_num_ + j] -= 1;
            ++weightsum;
          }
        }
      }
    }

    // Scale gradient

    Dtype normalizer = LossLayer<Dtype>::GetNormalizer(
        normalization_, outer_num_, inner_num_, weightsum);
    Dtype loss_weight = top[0]->cpu_diff()[0] / normalizer;
/*=======
    Dtype loss_weight = top[0]->cpu_diff()[0] /
                        get_normalizer(normalization_, weightsum);
>>>>>>> 83564bdf90c01da29daf42dfcc2bf7bce2f8a7fa*/
    caffe_scal(prob_.count(), loss_weight, bottom_diff);
  }
}

#ifdef CPU_ONLY
STUB_GPU(SoftmaxWithLossLayer);
#endif

INSTANTIATE_CLASS(SoftmaxWithLossLayer);
REGISTER_LAYER_CLASS(SoftmaxWithLoss);

}  // namespace caffe
