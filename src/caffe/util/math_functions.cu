#include <math_functions.h>  // CUDA's, not caffe's, for fabs, signbit
#include <thrust/device_vector.h>
#include <thrust/functional.h>  // thrust::plus
#include <thrust/reduce.h>

#include <cmath>

#include "caffe/common.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <>
void caffe_gpu_gemm<float>(const CBLAS_TRANSPOSE TransA,
    const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
    const float alpha, const float* A, const float* B, const float beta,
    float* C) {
  // Note that cublas follows fortran order.
  int lda = (TransA == CblasNoTrans) ? K : M;
  int ldb = (TransB == CblasNoTrans) ? N : K;
  cublasOperation_t cuTransA =
      (TransA == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
  cublasOperation_t cuTransB =
      (TransB == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
  CUBLAS_CHECK(cublasSgemm(Caffe::cublas_handle(), cuTransB, cuTransA,
      N, M, K, &alpha, B, ldb, A, lda, &beta, C, N));
}

template <>
void caffe_gpu_gemm<double>(const CBLAS_TRANSPOSE TransA,
    const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
    const double alpha, const double* A, const double* B, const double beta,
    double* C) {
  // Note that cublas follows fortran order.
  int lda = (TransA == CblasNoTrans) ? K : M;
  int ldb = (TransB == CblasNoTrans) ? N : K;
  cublasOperation_t cuTransA =
      (TransA == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
  cublasOperation_t cuTransB =
      (TransB == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
  CUBLAS_CHECK(cublasDgemm(Caffe::cublas_handle(), cuTransB, cuTransA,
      N, M, K, &alpha, B, ldb, A, lda, &beta, C, N));
}

// gemm sparse start
template <typename Dtype>
__global__ void matrix_multi_kernel(const int nnz, const int* rowInd, const int* colInd, const Dtype* val,
    const Dtype* B, const int colSizeB, Dtype* target) {
  CUDA_KERNEL_LOOP(index, nnz * colSizeB) {
    // ind is the index of val
    int ind = index / colSizeB;
    // col is the col if B
    int col = index % colSizeB;
    int ra = rowInd[ind];
    int ca = colInd[ind];
    target[ra * colSizeB + col] += val[ind] * B[ca * colSizeB + col];
  }
}

template <typename Dtype>
__global__ void matrix_multi_transA_kernel(const int nnz, const int* rowInd, const int* colInd,
    const Dtype* val, const Dtype* B, const int colSizeB, Dtype* target) {
  CUDA_KERNEL_LOOP(index, nnz * colSizeB) {
    // ind is the index of val
    int ind = index / colSizeB;
    // col is the col if B
    int col = index % colSizeB;
    int ra = colInd[ind];
    int ca = rowInd[ind];
    target[ra * colSizeB + col] += val[ind] * B[ca * colSizeB + col];
  }
}

template <>
void caffe_gpu_sparse_multi<float>(const CBLAS_TRANSPOSE TransA,
    const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
    const float* A, const int nnzB, const float alpha,
    const float* valB, const int* rowIndB, const int* colIndB,
    const float beta, float* C) {
  int ac = (TransA == CblasNoTrans) ? M : K;
  cudaMemset((void *)C, 0, M * N * sizeof(float));   
  //LOG(INFO) << "ac: " << ac;
  //LOG(INFO) << "M: " << M;
  if(TransB == CblasNoTrans) {
    matrix_multi_kernel<float><<<CAFFE_GET_BLOCKS(nnzB), CAFFE_CUDA_NUM_THREADS>>>(nnzB, rowIndB, colIndB, valB, A, ac, C);
  } else {
    matrix_multi_transA_kernel<float><<<CAFFE_GET_BLOCKS(nnzB), CAFFE_CUDA_NUM_THREADS>>>(nnzB, rowIndB, colIndB, valB, A, ac, C);
  }
}

template <>
void caffe_gpu_sparse_multi<double>(const CBLAS_TRANSPOSE TransA,
    const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
    const double* A, const int nnzB, const double alpha,
    const double* valB, const int* rowIndB, const int* colIndB,
    const double beta, double * C) {
  int ac = (TransA == CblasNoTrans) ? K : M;
  cudaMemset((void *)C, 0, M * N * sizeof(double));   
  if(TransB == CblasNoTrans) {
    matrix_multi_kernel<double><<<CAFFE_GET_BLOCKS(nnzB), CAFFE_CUDA_NUM_THREADS>>>(nnzB, rowIndB, colIndB, valB, A, ac, C);
  } else {
    matrix_multi_transA_kernel<double><<<CAFFE_GET_BLOCKS(nnzB), CAFFE_CUDA_NUM_THREADS>>>(nnzB, rowIndB, colIndB, valB, A, ac, C);
  }
}

template <>
void caffe_gpu_gemm_sparse<float>(const CBLAS_TRANSPOSE TransA,
    const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
    const float* A, const int nnzB, const float alpha,
    const float* csrValB, const int* csrRowPtrB, const int* csrColIndB,
    const float beta, float* C) {
  // Note that cublas follows fortran order.
  int lda = (TransA == CblasNoTrans) ? K : M;
  int br = (TransB == CblasNoTrans) ? N : K;
  int bc = (TransB == CblasNoTrans) ? K : N;
  cusparseOperation_t cuTransA =
      (TransA == CblasNoTrans) ? CUSPARSE_OPERATION_NON_TRANSPOSE : CUSPARSE_OPERATION_TRANSPOSE;
  cusparseOperation_t cuTransB =
      (TransB == CblasNoTrans) ? CUSPARSE_OPERATION_NON_TRANSPOSE : CUSPARSE_OPERATION_TRANSPOSE;
  CUSPARSE_CHECK(cusparseScsrmm2(Caffe::cusparse_handle(), cuTransB, cuTransA,
      br, M, bc, nnzB, &alpha,
      Caffe::cusparse_descr(), csrValB, csrRowPtrB, csrColIndB,
      A, lda, &beta, C, N));
}

template <>
void caffe_gpu_gemm_sparse<double>(const CBLAS_TRANSPOSE TransA,
    const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
    const double* A, const int nnzB, const double alpha,
    const double* csrValB, const int* csrRowPtrB, const int* csrColIndB,
    const double beta, double* C) {
  // Note that cublas follows fortran order.
  int lda = (TransA == CblasNoTrans) ? K : M;
  cusparseOperation_t cuTransA =
      (TransA == CblasNoTrans) ? CUSPARSE_OPERATION_NON_TRANSPOSE : CUSPARSE_OPERATION_TRANSPOSE;
  cusparseOperation_t cuTransB =
      (TransB == CblasNoTrans) ? CUSPARSE_OPERATION_NON_TRANSPOSE : CUSPARSE_OPERATION_TRANSPOSE;
  CUSPARSE_CHECK(cusparseDcsrmm2(Caffe::cusparse_handle(), cuTransB, cuTransA,
      N, M, K, nnzB, &alpha,
      Caffe::cusparse_descr(), csrValB, csrRowPtrB, csrColIndB,
      A, lda, &beta, C, N));
}
// gemm sparse end

template <>
void caffe_gpu_gemv<float>(const CBLAS_TRANSPOSE TransA, const int M,
    const int N, const float alpha, const float* A, const float* x,
    const float beta, float* y) {
  cublasOperation_t cuTransA =
      (TransA == CblasNoTrans) ? CUBLAS_OP_T : CUBLAS_OP_N;
  CUBLAS_CHECK(cublasSgemv(Caffe::cublas_handle(), cuTransA, N, M, &alpha,
      A, N, x, 1, &beta, y, 1));
}

template <>
void caffe_gpu_gemv<double>(const CBLAS_TRANSPOSE TransA, const int M,
    const int N, const double alpha, const double* A, const double* x,
    const double beta, double* y) {
  cublasOperation_t cuTransA =
      (TransA == CblasNoTrans) ? CUBLAS_OP_T : CUBLAS_OP_N;
  CUBLAS_CHECK(cublasDgemv(Caffe::cublas_handle(), cuTransA, N, M, &alpha,
      A, N, x, 1, &beta, y, 1));
}

// sparse
template <>
void caffe_gpu_gemv_sparse<float>(const CBLAS_TRANSPOSE TransA, const int M,
    const int N, const float alpha, const int nnzA,
    const float* csrValA, const int *csrRowPtrA, const int *csrColIndA,
    const float* x, const float beta, float* y) {
  cusparseOperation_t cuTransA =
      (TransA == CblasNoTrans) ? CUSPARSE_OPERATION_NON_TRANSPOSE:CUSPARSE_OPERATION_TRANSPOSE;
  CUSPARSE_CHECK(cusparseScsrmv(Caffe::cusparse_handle(), cuTransA, N, M, nnzA, &alpha,
      Caffe::cusparse_descr(), csrValA, csrRowPtrA, csrColIndA, x, &beta, y));
}

template <>
void caffe_gpu_gemv_sparse<double>(const CBLAS_TRANSPOSE TransA, const int M,
    const int N, const double alpha, const int nnzA,
    const double* csrValA, const int *csrRowPtrA, const int *csrColIndA,
    const double* x, const double beta, double* y) {
  cusparseOperation_t cuTransA =
      (TransA == CblasNoTrans) ? CUSPARSE_OPERATION_NON_TRANSPOSE:CUSPARSE_OPERATION_TRANSPOSE;
  CUSPARSE_CHECK(cusparseDcsrmv(Caffe::cusparse_handle(), cuTransA, N, M, nnzA, &alpha,
      Caffe::cusparse_descr(), csrValA, csrRowPtrA, csrColIndA, x, &beta, y));
}
// end sparse

template <>
void caffe_gpu_axpy<float>(const int N, const float alpha, const float* X,
    float* Y) {
  CUBLAS_CHECK(cublasSaxpy(Caffe::cublas_handle(), N, &alpha, X, 1, Y, 1));
}

template <>
void caffe_gpu_axpy<double>(const int N, const double alpha, const double* X,
    double* Y) {
  CUBLAS_CHECK(cublasDaxpy(Caffe::cublas_handle(), N, &alpha, X, 1, Y, 1));
}

void caffe_gpu_memcpy(const size_t N, const void* X, void* Y) {
  if (X != Y) {
    CUDA_CHECK(cudaMemcpy(Y, X, N, cudaMemcpyDefault));  // NOLINT(caffe/alt_fn)
  }
}

template <>
void caffe_gpu_scal<float>(const int N, const float alpha, float *X) {
  CUBLAS_CHECK(cublasSscal(Caffe::cublas_handle(), N, &alpha, X, 1));
}

template <>
void caffe_gpu_scal<double>(const int N, const double alpha, double *X) {
  CUBLAS_CHECK(cublasDscal(Caffe::cublas_handle(), N, &alpha, X, 1));
}

template <>
void caffe_gpu_scal<float>(const int N, const float alpha, float* X,
                           cudaStream_t str) {
  cudaStream_t initial_stream;
  CUBLAS_CHECK(cublasGetStream(Caffe::cublas_handle(), &initial_stream));
  CUBLAS_CHECK(cublasSetStream(Caffe::cublas_handle(), str));
  CUBLAS_CHECK(cublasSscal(Caffe::cublas_handle(), N, &alpha, X, 1));
  CUBLAS_CHECK(cublasSetStream(Caffe::cublas_handle(), initial_stream));
}

template <>
void caffe_gpu_scal<double>(const int N, const double alpha, double* X,
                            cudaStream_t str) {
  cudaStream_t initial_stream;
  CUBLAS_CHECK(cublasGetStream(Caffe::cublas_handle(), &initial_stream));
  CUBLAS_CHECK(cublasSetStream(Caffe::cublas_handle(), str));
  CUBLAS_CHECK(cublasDscal(Caffe::cublas_handle(), N, &alpha, X, 1));
  CUBLAS_CHECK(cublasSetStream(Caffe::cublas_handle(), initial_stream));
}

template <>
void caffe_gpu_axpby<float>(const int N, const float alpha, const float* X,
    const float beta, float* Y) {
  caffe_gpu_scal<float>(N, beta, Y);
  caffe_gpu_axpy<float>(N, alpha, X, Y);
}

template <>
void caffe_gpu_axpby<double>(const int N, const double alpha, const double* X,
    const double beta, double* Y) {
  caffe_gpu_scal<double>(N, beta, Y);
  caffe_gpu_axpy<double>(N, alpha, X, Y);
}

template <>
void caffe_gpu_dot<float>(const int n, const float* x, const float* y,
    float* out) {
  CUBLAS_CHECK(cublasSdot(Caffe::cublas_handle(), n, x, 1, y, 1, out));
}

template <>
void caffe_gpu_dot<double>(const int n, const double* x, const double* y,
    double * out) {
  CUBLAS_CHECK(cublasDdot(Caffe::cublas_handle(), n, x, 1, y, 1, out));
}

template <>
void caffe_gpu_asum<float>(const int n, const float* x, float* y) {
  CUBLAS_CHECK(cublasSasum(Caffe::cublas_handle(), n, x, 1, y));
}

template <>
void caffe_gpu_asum<double>(const int n, const double* x, double* y) {
  CUBLAS_CHECK(cublasDasum(Caffe::cublas_handle(), n, x, 1, y));
}

template <>
void caffe_gpu_scale<float>(const int n, const float alpha, const float *x,
                            float* y) {
  CUBLAS_CHECK(cublasScopy(Caffe::cublas_handle(), n, x, 1, y, 1));
  CUBLAS_CHECK(cublasSscal(Caffe::cublas_handle(), n, &alpha, y, 1));
}

template <>
void caffe_gpu_scale<double>(const int n, const double alpha, const double *x,
                             double* y) {
  CUBLAS_CHECK(cublasDcopy(Caffe::cublas_handle(), n, x, 1, y, 1));
  CUBLAS_CHECK(cublasDscal(Caffe::cublas_handle(), n, &alpha, y, 1));
}

template <typename Dtype>
__global__ void set_kernel(const int n, const Dtype alpha, Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = alpha;
  }
}

template <typename Dtype>
void caffe_gpu_set(const int N, const Dtype alpha, Dtype* Y) {
  if (alpha == 0) {
    CUDA_CHECK(cudaMemset(Y, 0, sizeof(Dtype) * N));  // NOLINT(caffe/alt_fn)
    return;
  }
  // NOLINT_NEXT_LINE(whitespace/operators)
  set_kernel<Dtype><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, alpha, Y);
}

template void caffe_gpu_set<int>(const int N, const int alpha, int* Y);
template void caffe_gpu_set<float>(const int N, const float alpha, float* Y);
template void caffe_gpu_set<double>(const int N, const double alpha, double* Y);

template <typename Dtype>
__global__ void add_scalar_kernel(const int n, const Dtype alpha, Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] += alpha;
  }
}

template <>
void caffe_gpu_add_scalar(const int N, const float alpha, float* Y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  add_scalar_kernel<float><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, alpha, Y);
}

template <>
void caffe_gpu_add_scalar(const int N, const double alpha, double* Y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  add_scalar_kernel<double><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, alpha, Y);
}

template <typename Dtype>
__global__ void add_kernel(const int n, const Dtype* a,
    const Dtype* b, Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = a[index] + b[index];
  }
}

template <>
void caffe_gpu_add<float>(const int N, const float* a, const float* b,
    float* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  add_kernel<float><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, b, y);
}

template <>
void caffe_gpu_add<double>(const int N, const double* a, const double* b,
    double* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  add_kernel<double><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, b, y);
}

template <typename Dtype>
__global__ void sub_kernel(const int n, const Dtype* a,
    const Dtype* b, Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = a[index] - b[index];
  }
}

template <>
void caffe_gpu_sub<float>(const int N, const float* a, const float* b,
    float* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  sub_kernel<float><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, b, y);
}

template <>
void caffe_gpu_sub<double>(const int N, const double* a, const double* b,
    double* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  sub_kernel<double><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, b, y);
}

template <typename Dtype>
__global__ void mul_kernel(const int n, const Dtype* a,
    const Dtype* b, Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = a[index] * b[index];
  }
}

template <>
void caffe_gpu_mul<float>(const int N, const float* a,
    const float* b, float* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  mul_kernel<float><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, b, y);
}

template <>
void caffe_gpu_mul<double>(const int N, const double* a,
    const double* b, double* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  mul_kernel<double><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, b, y);
}

template <typename Dtype>
__global__ void div_kernel(const int n, const Dtype* a,
    const Dtype* b, Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = a[index] / b[index];
  }
}

template <>
void caffe_gpu_div<float>(const int N, const float* a,
    const float* b, float* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  div_kernel<float><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, b, y);
}

template <>
void caffe_gpu_div<double>(const int N, const double* a,
    const double* b, double* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  div_kernel<double><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, b, y);
}

template <typename Dtype>
__global__ void abs_kernel(const int n, const Dtype* a, Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = abs(a[index]);
  }
}

template <>
void caffe_gpu_abs<float>(const int N, const float* a, float* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  abs_kernel<float><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, y);
}

template <>
void caffe_gpu_abs<double>(const int N, const double* a, double* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  abs_kernel<double><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, y);
}


template <typename Dtype>
__global__ void exp_kernel(const int n, const Dtype* a, Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = exp(a[index]);
  }
}

template <>
void caffe_gpu_exp<float>(const int N, const float* a, float* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  exp_kernel<float><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, y);
}

template <>
void caffe_gpu_exp<double>(const int N, const double* a, double* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  exp_kernel<double><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, y);
}

template <typename Dtype>
__global__ void log_kernel(const int n, const Dtype* a, Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = log(a[index]);
  }
}

template <>
void caffe_gpu_log<float>(const int N, const float* a, float* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  log_kernel<float><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, y);
}

template <>
void caffe_gpu_log<double>(const int N, const double* a, double* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  log_kernel<double><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, y);
}

template <typename Dtype>
__global__ void powx_kernel(const int n, const Dtype* a,
    const Dtype alpha, Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = pow(a[index], alpha);
  }
}

template <>
void caffe_gpu_powx<float>(const int N, const float* a,
    const float alpha, float* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  powx_kernel<float><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, alpha, y);
}

template <>
void caffe_gpu_powx<double>(const int N, const double* a,
    const double alpha, double* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  powx_kernel<double><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, alpha, y);
}

template <typename Dtype>
__global__ void sqrt_kernel(const int n, const Dtype* a, Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = sqrt(a[index]);
  }
}

template <>
void caffe_gpu_sqrt<float>(const int N, const float* a, float* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  sqrt_kernel<float><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, y);
}

template <>
void caffe_gpu_sqrt<double>(const int N, const double* a, double* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  sqrt_kernel<double><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, y);
}

DEFINE_AND_INSTANTIATE_GPU_UNARY_FUNC(sign, y[index] = (Dtype(0) < x[index])
                                      - (x[index] < Dtype(0)));
DEFINE_AND_INSTANTIATE_GPU_UNARY_FUNC(sgnbit, y[index] = signbit(x[index]));

void caffe_gpu_rng_uniform(const int n, unsigned int* r) {
  CURAND_CHECK(curandGenerate(Caffe::curand_generator(), r, n));
}

template <>
void caffe_gpu_rng_uniform<float>(const int n, const float a, const float b,
                                  float* r) {
  CURAND_CHECK(curandGenerateUniform(Caffe::curand_generator(), r, n));
  const float range = b - a;
  if (range != static_cast<float>(1)) {
    caffe_gpu_scal(n, range, r);
  }
  if (a != static_cast<float>(0)) {
    caffe_gpu_add_scalar(n, a, r);
  }
}

template <>
void caffe_gpu_rng_uniform<double>(const int n, const double a, const double b,
                                   double* r) {
  CURAND_CHECK(curandGenerateUniformDouble(Caffe::curand_generator(), r, n));
  const double range = b - a;
  if (range != static_cast<double>(1)) {
    caffe_gpu_scal(n, range, r);
  }
  if (a != static_cast<double>(0)) {
    caffe_gpu_add_scalar(n, a, r);
  }
}

template <>
void caffe_gpu_rng_gaussian(const int n, const float mu, const float sigma,
                            float* r) {
  CURAND_CHECK(
      curandGenerateNormal(Caffe::curand_generator(), r, n, mu, sigma));
}

template <>
void caffe_gpu_rng_gaussian(const int n, const double mu, const double sigma,
                            double* r) {
  CURAND_CHECK(
      curandGenerateNormalDouble(Caffe::curand_generator(), r, n, mu, sigma));
}

template <typename Dtype>
__global__ void sparse_add_kernel(const int colSize, const int nnz, const Dtype* denseVal, const int* rowInd, const int* colInd, Dtype* y) {
  CUDA_KERNEL_LOOP(index, nnz) {
    y[index] -= denseVal[rowInd[index] * colSize + colInd[index]];
  }
}

template <>
void caffe_gpu_sparse_add<float>(const int rowSize, const int colSize, const int nnz, float* val, const int* rowInd, const int* colInd, const float* denseVal) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  //LOG(INFO) << "before y[0]: " << val[0] << " diff: " << denseVal[colInd[0] * rowSize + rowInd[0]];
  sparse_add_kernel<float><<<CAFFE_GET_BLOCKS(nnz), CAFFE_CUDA_NUM_THREADS>>>(
      colSize, nnz, denseVal, rowInd, colInd, val);
  //LOG(INFO) << "after y[0]: " << val[0];
}

void caffe_gpu_csr2coo(const int* rowPtr, int nnz, int rowSize, int *rowInd) {
  cusparseXcsr2coo(Caffe::cusparse_handle(), rowPtr, nnz, rowSize, rowInd, CUSPARSE_INDEX_BASE_ZERO);
}

template <>
void caffe_gpu_sparse_add<double>(const int rowSize, const int colSize, const int nnz, double* val, const int* rowInd, const int* colInd, const double* denseVal) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  sparse_add_kernel<double><<<CAFFE_GET_BLOCKS(nnz), CAFFE_CUDA_NUM_THREADS>>>(
      colSize, nnz, denseVal, rowInd, colInd, val);
  //cudaDeviceSynchronize();
}

template<>
void caffe_dense2sparse<float>(int m, int n, const float* A, float percentage,
        int *nnz, float** val, int** colInd, int** rowPtr) {
  cudaStream_t stream = NULL;
  int *d_csrRowPtrC = NULL;
  int *d_csrColIndC = NULL;
  float *d_csrValC = NULL;
  char *d_work= NULL;
  pruneInfo_t info = NULL;
  size_t lworkInBytes = 0;
  cusparseCreatePruneInfo(&info);
  cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
  cusparseSetStream(Caffe::cusparse_handle(), stream);

  cudaMalloc((void**)&d_csrRowPtrC, sizeof(int)*(m+1));
  CUSPARSE_CHECK(cusparseSpruneDense2csrByPercentage_bufferSizeExt(
    Caffe::cusparse_handle(),
    m, n, A, m, percentage,
    Caffe::cusparse_descr(),
    d_csrValC,
    d_csrRowPtrC,
    d_csrColIndC,
    info, &lworkInBytes
  ));
  cudaMalloc((void**)&d_work, lworkInBytes);
  CUSPARSE_CHECK(cusparseSpruneDense2csrNnzByPercentage(
    Caffe::cusparse_handle(),
    m, n, A, m, percentage,
    Caffe::cusparse_descr(),
    d_csrRowPtrC,
    nnz, info, d_work
  ));
  cudaMalloc((void**)&d_csrColIndC, sizeof(int) * (*nnz));
  cudaMalloc((void**)&d_csrValC, sizeof(float) * (*nnz));
  CUSPARSE_CHECK(cusparseSpruneDense2csrByPercentage(
    Caffe::cusparse_handle(),
    m, n, A, m, percentage,
    Caffe::cusparse_descr(),
    d_csrValC,
    d_csrRowPtrC,
    d_csrColIndC,
    info, d_work
  ));
  *rowPtr = (int*) malloc(sizeof(int )*(m + 1));
  *colInd = (int*) malloc(sizeof(int )* (*nnz));
  *val = (float*) malloc(sizeof(float)* (*nnz));

  cudaMemcpy(*rowPtr, d_csrRowPtrC, sizeof(int)*(m+1), cudaMemcpyDeviceToHost);
  cudaMemcpy(*colInd, d_csrColIndC, sizeof(int)* (*nnz), cudaMemcpyDeviceToHost);
  cudaMemcpy(*val, d_csrValC , sizeof(float)* (*nnz), cudaMemcpyDeviceToHost);

  if (d_csrRowPtrC) cudaFree(d_csrRowPtrC);
  if (d_csrColIndC) cudaFree(d_csrColIndC);
  if (d_csrValC) cudaFree(d_csrValC);
  if (stream) cudaStreamDestroy(stream);

  if(info) cusparseDestroyPruneInfo(info);
}

template<>
void caffe_dense2sparse<double>(int m, int n, const double* A, float percentage,
        int *nnz, double** val, int** colInd, int** rowPtr) {
  cudaStream_t stream = NULL;
  int *d_csrRowPtrC = NULL;
  int *d_csrColIndC = NULL;
  double*d_csrValC = NULL;
  char *d_work= NULL;
  pruneInfo_t info = NULL;
  size_t lworkInBytes = 0;
  cusparseCreatePruneInfo(&info);
  cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
  cusparseSetStream(Caffe::cusparse_handle(), stream);

  cudaMalloc((void**)&d_csrRowPtrC, sizeof(int)*(m+1));
  CUSPARSE_CHECK(cusparseDpruneDense2csrByPercentage_bufferSizeExt(
    Caffe::cusparse_handle(),
    m, n, A, m, percentage,
    Caffe::cusparse_descr(),
    d_csrValC,
    d_csrRowPtrC,
    d_csrColIndC,
    info, &lworkInBytes
  ));
  cudaMalloc((void**)&d_work, lworkInBytes);
  CUSPARSE_CHECK(cusparseDpruneDense2csrNnzByPercentage(
    Caffe::cusparse_handle(),
    m, n, A, m, percentage,
    Caffe::cusparse_descr(),
    d_csrRowPtrC,
    nnz, info, d_work
  ));
  cudaDeviceSynchronize();
  cudaMalloc((void**)&d_csrColIndC, sizeof(int ) * (*nnz));
  cudaMalloc((void**)&d_csrValC , sizeof(double) * (*nnz));
  CUSPARSE_CHECK(cusparseDpruneDense2csrByPercentage(
    Caffe::cusparse_handle(),
    m, n, A, m, percentage,
    Caffe::cusparse_descr(),
    d_csrValC,
    d_csrRowPtrC,
    d_csrColIndC,
    info, d_work
  ));
  cudaDeviceSynchronize();
  *rowPtr = (int* )malloc(sizeof(int )*(m + 1));
  *colInd = (int* )malloc(sizeof(int )* (*nnz));
  *val = (double*)malloc(sizeof(double)* (*nnz));

  cudaMemcpy(*rowPtr, d_csrRowPtrC, sizeof(int)*(m+1), cudaMemcpyDeviceToHost);
  cudaMemcpy(*colInd, d_csrColIndC, sizeof(int)* (*nnz), cudaMemcpyDeviceToHost);
  cudaMemcpy(*val, d_csrValC , sizeof(double)* (*nnz), cudaMemcpyDeviceToHost);

  if (d_csrRowPtrC) cudaFree(d_csrRowPtrC);
  if (d_csrColIndC) cudaFree(d_csrColIndC);
  if (d_csrValC) cudaFree(d_csrValC);
  if (stream) cudaStreamDestroy(stream);

  if(info) cusparseDestroyPruneInfo(info);
}

}  // namespace caffe
