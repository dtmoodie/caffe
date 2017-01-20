#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/layer_factory.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/layers/seginfogain_loss_layer.hpp"

namespace caffe {

	template <typename Dtype>
	void SegInfogainLossLayer<Dtype>::LayerSetUp(
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

		if (bottom.size() < 3) {
			CHECK(this->layer_param_.infogain_loss_param().has_source())
				<< "Infogain matrix source must be specified.";
			BlobProto blob_proto;
			ReadProtoFromBinaryFile(
				this->layer_param_.infogain_loss_param().source(), &blob_proto);
			infogain_.FromProto(blob_proto);

			infogain_sum_.Reshape(infogain_.shape()); // We could reduce this by a dimension for efficiency
			Dtype* infogain_sum = infogain_sum_.mutable_cpu_data();
			for (int labelIt = 0; labelIt < infogain_.shape(3); labelIt++)
			{
				infogain_sum[labelIt] = caffe_cpu_asum(infogain_.shape(3), infogain_.cpu_data() + labelIt * infogain_.shape(3));
			}

		}

	}

	template <typename Dtype>
	void SegInfogainLossLayer<Dtype>::Reshape(
		const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
		LossLayer<Dtype>::Reshape(bottom, top);
		Blob<Dtype>* infogain = NULL;
		if (bottom.size() < 3) {
			infogain = &infogain_;
		}
		else {
			infogain = bottom[2];
		}
		infogain_sum_.Reshape(infogain->shape()); // We could reduce this by a dimension for efficiency

		softmax_layer_->Reshape(softmax_bottom_vec_, softmax_top_vec_);
		softmax_axis_ =
			bottom[0]->CanonicalAxisIndex(this->layer_param_.softmax_param().axis());
		outer_num_ = bottom[0]->count(0, softmax_axis_); // # cases?
		inner_num_ = bottom[0]->count(softmax_axis_ + 1); // # voxels?
		CHECK_EQ(outer_num_ * inner_num_, bottom[1]->count())
			<< "Number of labels must match number of predictions; "
			<< "e.g., if softmax axis == 1 and prediction shape is (N, C, H, W), "
			<< "label count (number of labels) must be N*H*W, "
			<< "with integer values in {0, 1, ..., C-1}.";
	}

	template <typename Dtype>
	void SegInfogainLossLayer<Dtype>::Forward_cpu(
		const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
		// The forward pass computes the softmax prob values.
		softmax_layer_->Forward(softmax_bottom_vec_, softmax_top_vec_);
		const Dtype* prob_data = prob_.cpu_data();
		const Dtype* label = bottom[1]->cpu_data();


		const Dtype* infogain_mat = NULL;
		if (bottom.size() < 3) {
			infogain_mat = infogain_.cpu_data();
		}
		else {
			infogain_mat = bottom[2]->cpu_data();
		}

		int dim = prob_.count() / outer_num_; //step between cases: number of voxels*labels
		int numLabels = prob_.count() / outer_num_ / inner_num_;
		int count = 0;
		Dtype loss = 0;
		for (int i = 0; i < outer_num_; ++i) { // for each case
			for (int j = 0; j < inner_num_; j++) { // for each voxel
				const int label_value = static_cast<int>(label[i * inner_num_ + j]);
				if (has_ignore_label_ && label_value == ignore_label_) {
					continue;
				}
				DCHECK_GE(label_value, 0);
				DCHECK_LT(label_value, prob_.shape(softmax_axis_));
				for (int k = 0; k < numLabels; k++)
				{
                    loss -= infogain_mat[label_value * numLabels + k] *
                            log(std::max(prob_data[i * dim + k * inner_num_ + j],
                                Dtype(kLOG_THRESHOLD)));
				}
				++count;

			}
		}
		if (normalize_) {
			top[0]->mutable_cpu_data()[0] = loss / count;
		}
		else {
			top[0]->mutable_cpu_data()[0] = loss / outer_num_;
		}
		if (top.size() == 2) {
			top[1]->ShareData(prob_);
		}
	}

	template <typename Dtype>
	void SegInfogainLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
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
			Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
			const Dtype* prob_data = prob_.cpu_data();

			caffe_copy(prob_.count(), prob_data, bottom_diff);
			const Dtype* label = bottom[1]->cpu_data();
			const Dtype* infogain_mat = NULL;
			Dtype* infogain_sum = infogain_sum_.mutable_cpu_data();

			int numLabels = prob_.count() / outer_num_ / inner_num_;
			int dim = prob_.count() / outer_num_;
			if (bottom.size() < 3) {
				infogain_mat = infogain_.cpu_data();
			}
			else {
				infogain_mat = bottom[2]->cpu_data();
				for (int labelIt = 0; labelIt < numLabels; labelIt++)
				{
					infogain_sum[labelIt] = caffe_cpu_asum(numLabels, infogain_mat + labelIt * numLabels);

				}
			}


			int count = 0;
			for (int i = 0; i < outer_num_; ++i) {
				for (int j = 0; j < inner_num_; ++j) {
					const int label_value = static_cast<int>(label[i * inner_num_ + j]);
					if (has_ignore_label_ && label_value == ignore_label_) {
						for (int c = 0; c < bottom[0]->shape(softmax_axis_); ++c) {
							bottom_diff[i * dim + c * inner_num_ + j] = 0;
						}
					}
					else {

						for (int k = 0; k < numLabels; k++)
						{
							bottom_diff[i * dim + k * inner_num_ + j] *= infogain_sum[label_value];
							bottom_diff[i * dim + k * inner_num_ + j] -= infogain_mat[label_value * numLabels + k];
						}
						++count;
					}
				}
			}

			// Scale gradient
			const Dtype loss_weight = top[0]->cpu_diff()[0];
            if (normalize_) {
                caffe_scal(prob_.count(), loss_weight / count, bottom_diff);
			}
			else {
                caffe_scal(prob_.count(), loss_weight / outer_num_, bottom_diff);
			}
		}
	}

#ifdef CPU_ONLY
	STUB_GPU(SegInfogainLossLayer);
#endif

	INSTANTIATE_CLASS(SegInfogainLossLayer);
	REGISTER_LAYER_CLASS(SegInfogainLoss);

}  // namespace caffe
