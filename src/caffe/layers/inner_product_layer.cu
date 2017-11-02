#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/inner_product_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void InnerProductLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  const Dtype* weight = this->blobs_[0]->gpu_data();
  bool sparse = this->sparse_;
  int nnzW = this->blobs_[0]->nnz();
  const Dtype* csrValW = this->blobs_[0]->gpu_val();
  const int* csrRowPtrW = this->blobs_[0]->gpu_row_ptr();
  const int* csrColIndW = this->blobs_[0]->gpu_col_ind();
  if (M_ == 1) {
    if(sparse) {
      caffe_gpu_gemv_sparse<Dtype>(CblasNoTrans, N_, K_, (Dtype)1.,
                         nnzW, csrValW, csrRowPtrW, csrColIndW, bottom_data,
                         (Dtype)0., top_data);
    } else {
      caffe_gpu_gemv<Dtype>(CblasNoTrans, N_, K_, (Dtype)1.,
                         weight, bottom_data, (Dtype)0., top_data);
    }
    if (bias_term_)
      caffe_gpu_axpy<Dtype>(N_, bias_multiplier_.cpu_data()[0],
                            this->blobs_[1]->gpu_data(), top_data);
  } else {
    //bias_multiplier_.gpu_data();
    //LOG(INFO) << "bias done";
    //this->blobs_[1]->gpu_data();
    //LOG(INFO) << "blobs done";
    //LOG(INFO) << "nnzw: " << nnzW;
    //LOG(INFO) << "weight shape: " << this->blobs_[0]->shape_string();
    //LOG(INFO) << "top shape: " << top[0]->shape_string();
    //LOG(INFO) << "bottom shape: " << bottom[0]->shape_string();
    //LOG(INFO) << "row[nnz-1]: " << this->blobs_[0]->cpu_row_ind()[nnzW-1];
    //LOG(INFO) << "col[0]: " << this->blobs_[0]->cpu_col_ind()[0];
    //LOG(INFO) << "top[0]: " << top[0]->cpu_data()[0];
    //LOG(INFO) << "M: " << M_;
    //LOG(INFO) << "N: " << N_;
    //LOG(INFO) << "K: " << K_;
    //clock_t start = clock();
    if(sparse) {
      //const int* csrRowIndW = this->blobs_[0]->gpu_row_ind();
      caffe_gpu_gemm_sparse<Dtype>(CblasNoTrans,
                          transpose_ ? CblasNoTrans : CblasTrans,
                          M_, N_, K_, bottom_data,
                          nnzW, (Dtype)1., csrValW, csrRowPtrW, csrColIndW,
                          (Dtype)0., top_data);
      //caffe_gpu_sparse_multi(CblasNoTrans,
      //                    transpose_ ? CblasNoTrans : CblasTrans,
      //                    M_, N_, K_, bottom_data,
      //                    nnzW, (Dtype)1., csrValW, csrRowIndW, csrColIndW,
      //                    (Dtype)0., top_data);
    } else {
      caffe_gpu_gemm<Dtype>(CblasNoTrans,
                          transpose_ ? CblasNoTrans : CblasTrans,
                          M_, N_, K_, (Dtype)1.,
                          bottom_data, weight, (Dtype)0., top_data);
    }
    //clock_t ends = clock();
    //LOG(INFO) <<"Running Time : "<<(double)(ends - start)/ CLOCKS_PER_SEC;
    //LOG(INFO) << "wx done";
    //LOG(INFO) << "row[0]: " << this->blobs_[0]->cpu_row_ptr()[0];
    //LOG(INFO) << "col[0]: " << this->blobs_[0]->cpu_col_ind()[0];
    //LOG(INFO) << "top[0]: " << top[0]->cpu_data()[0];
    //bias_multiplier_.gpu_data();
    //cudaDeviceSynchronize();
    //LOG(INFO) << "bias done";
    //this->blobs_[1]->gpu_data();
    //cudaDeviceSynchronize();
    //LOG(INFO) << "blobs done";
    if (bias_term_)
      caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, N_, 1, (Dtype)1.,
                            bias_multiplier_.gpu_data(),
                            this->blobs_[1]->gpu_data(), (Dtype)1., top_data);
  }
}

//template <typename Dtype>
//void InnerProductLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
//    const vector<Blob<Dtype>*>& top) {
//  const Dtype* bottom_data = bottom[0]->gpu_data();
//  Dtype* top_data = top[0]->mutable_gpu_data();
//  const Dtype* weight = this->blobs_[0]->gpu_data();
//  if (M_ == 1) {
//    caffe_gpu_gemv<Dtype>(CblasNoTrans, N_, K_, (Dtype)1.,
//                         weight, bottom_data, (Dtype)0., top_data);
//    if (bias_term_)
//      caffe_gpu_axpy<Dtype>(N_, bias_multiplier_.cpu_data()[0],
//                            this->blobs_[1]->gpu_data(), top_data);
//  } else {
//    caffe_gpu_gemm<Dtype>(CblasNoTrans,
//                          transpose_ ? CblasNoTrans : CblasTrans,
//                          M_, N_, K_, (Dtype)1.,
//                          bottom_data, weight, (Dtype)0., top_data);
//    if (bias_term_)
//      caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, N_, 1, (Dtype)1.,
//                            bias_multiplier_.gpu_data(),
//                            this->blobs_[1]->gpu_data(), (Dtype)1., top_data);
//  }
//}

template <typename Dtype>
void InnerProductLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  bool sparse = this->sparse_;
  if (this->param_propagate_down_[0]) {
    const Dtype* top_diff = top[0]->gpu_diff();
    const Dtype* bottom_data = bottom[0]->gpu_data();
    // Gradient with respect to weight
    if (transpose_) {
      caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans,
          K_, N_, M_,
          (Dtype)1., bottom_data, top_diff,
          (Dtype)1., this->blobs_[0]->mutable_gpu_diff());
    } else {
      caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans,
          N_, K_, M_,
          (Dtype)1., top_diff, bottom_data,
          (Dtype)1., this->blobs_[0]->mutable_gpu_diff());
    }
  }
  if (bias_term_ && this->param_propagate_down_[1]) {
    const Dtype* top_diff = top[0]->gpu_diff();
    // Gradient with respect to bias
    caffe_gpu_gemv<Dtype>(CblasTrans, M_, N_, (Dtype)1., top_diff,
        bias_multiplier_.gpu_data(), (Dtype)1.,
        this->blobs_[1]->mutable_gpu_diff());
  }
    //clock_t start = clock();
  if (propagate_down[0]) {
    const Dtype* top_diff = top[0]->gpu_diff();
    int nnzW = this->blobs_[0]->nnz();
    const Dtype* csrValW = this->blobs_[0]->gpu_val();
    const int* csrRowPtrW = this->blobs_[0]->gpu_row_ptr();
    const int* csrColIndW = this->blobs_[0]->gpu_col_ind();
    // Gradient with respect to bottom data
    //if (transpose_) {
    if(Caffe::step() == 2 && sparse) {
      //const int* csrRowIndW = this->blobs_[0]->gpu_row_ind();
      caffe_gpu_gemm_sparse<Dtype>(CblasNoTrans, transpose_ ? CblasTrans : CblasNoTrans,
          M_, K_, N_, top_diff,
          nnzW, (Dtype)1., csrValW, csrRowPtrW, csrColIndW,
          (Dtype)0., bottom[0]->mutable_gpu_diff());
      //caffe_gpu_sparse_multi<Dtype>(CblasNoTrans, transpose_ ? CblasTrans : CblasNoTrans,
      //    M_, K_, N_, top_diff,
      //    nnzW, (Dtype)1., csrValW, csrRowIndW, csrColIndW,
      //    (Dtype)0., bottom[0]->mutable_gpu_diff());
    } else {
      caffe_gpu_gemm<Dtype>(CblasNoTrans, transpose_ ? CblasTrans : CblasNoTrans,
          M_, K_, N_,
          (Dtype)1., top_diff, this->blobs_[0]->gpu_data(),
          (Dtype)0., bottom[0]->mutable_gpu_diff());
    }
  }
    //clock_t ends = clock();
    //LOG(INFO) <<"backward Running Time : "<<(double)(ends - start)/ CLOCKS_PER_SEC;
}

//template <typename Dtype>
//void InnerProductLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
//    const vector<bool>& propagate_down,
//    const vector<Blob<Dtype>*>& bottom) {
//  if (this->param_propagate_down_[0]) {
//    const Dtype* top_diff = top[0]->gpu_diff();
//    const Dtype* bottom_data = bottom[0]->gpu_data();
//    // Gradient with respect to weight
//    if (transpose_) {
//      caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans,
//          K_, N_, M_,
//          (Dtype)1., bottom_data, top_diff,
//          (Dtype)1., this->blobs_[0]->mutable_gpu_diff());
//    } else {
//      caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans,
//          N_, K_, M_,
//          (Dtype)1., top_diff, bottom_data,
//          (Dtype)1., this->blobs_[0]->mutable_gpu_diff());
//    }
//  }
//  if (bias_term_ && this->param_propagate_down_[1]) {
//    const Dtype* top_diff = top[0]->gpu_diff();
//    // Gradient with respect to bias
//    caffe_gpu_gemv<Dtype>(CblasTrans, M_, N_, (Dtype)1., top_diff,
//        bias_multiplier_.gpu_data(), (Dtype)1.,
//        this->blobs_[1]->mutable_gpu_diff());
//  }
//  if (propagate_down[0]) {
//    const Dtype* top_diff = top[0]->gpu_diff();
//    // Gradient with respect to bottom data
//    if (transpose_) {
//      caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans,
//          M_, K_, N_,
//          (Dtype)1., top_diff, this->blobs_[0]->gpu_data(),
//          (Dtype)0., bottom[0]->mutable_gpu_diff());
//    } else {
//      caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans,
//          M_, K_, N_,
//         (Dtype)1., top_diff, this->blobs_[0]->gpu_data(),
//         (Dtype)0., bottom[0]->mutable_gpu_diff());
//    }
//  }
//}

INSTANTIATE_LAYER_GPU_FUNCS(InnerProductLayer);

}  // namespace caffe
