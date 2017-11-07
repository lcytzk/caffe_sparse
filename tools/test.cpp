#include "caffe/caffe.hpp"

using namespace caffe;
using caffe::Caffe;

int main () {
    int c = 1000;
    int r = 1000;
    int lda = r;
    float A[1000000];// = {1, 0, 5, 0, 0, 4, 0, 8, 2, 0, 6, 0, -3, 0, 7, 9};
    float B[1000000];// = {1, 0, 5, 0, 0, 4, 0, 8, 2, 0, 6, 0, -3, 0, 7, 9};
    for(int i = 0; i < 1000000; ++i) {
    A[i] = i;
    B[i] = i;
    }
    float *C = NULL;
    float *d_A = NULL;
    cudaMalloc((void**)&d_A, sizeof(float)*lda*c);
    //cudaMemcpy(d_A, A, sizeof(float)*lda*c, cudaMemcpyHostToDevice);
    caffe_gpu_memcpy(sizeof(float)*lda*c, A, d_A);
    int nnz;
    int* rowPtr;
    int* colInd;
    float* val;
    caffe_dense2sparse(r, c, d_A, 99, &nnz, &val, &colInd, &rowPtr);

    int msize = c * r * sizeof(float);

    cudaMalloc(&C, msize);
    float *A_;
    cudaMalloc(&A_, msize);
    caffe_gpu_memcpy(msize, A, A_);

    float *B_;
    cudaMalloc(&B_, msize);
    caffe_gpu_memcpy(msize, B, B_);

    size_t start = clock();
    size_t end = clock();


    start = clock();
    for(int i = 0; i < 100; ++i) {
      caffe_gpu_gemm<float>(CblasNoTrans,
                          CblasNoTrans,
                          c, c, c, (float)1.,
                          A_, B_, (float)0., C);
      cudaDeviceSynchronize();
    }
    end = clock();
    printf("dense*dense no trans time used: %f\n", (double)(end-start)/CLOCKS_PER_SEC);

    start = clock();
    for(int i = 0; i < 100; ++i) {
      caffe_gpu_gemm<float>(CblasNoTrans,
                          CblasTrans,
                          c, c, c, (float)1.,
                          A_, B_, (float)0., C);
      cudaDeviceSynchronize();
    }
    end = clock();
    printf("dense*dense trans time used: %f\n", (double)(end-start)/CLOCKS_PER_SEC);

    return 0;
}
