#include "caffe/util/math_functions.hpp"

int main () {
    int c = 4;
    int r = 4;
    int lda = r;
    const float A[c][lda] = {1, 0, 5, 0, 0, 4, 0, 8, 2, 0, 6, 0, -3, 0, 7, 9};
    float *d_A = NULL;
    cudaMalloc((void**)&d_A, sizeof(float)*lda*c);
    cudaMemcpy(d_A, A, sizeof(float)*lda*c, cudaMemcpyHostToDevice);
    int nnz;
    int* rowPtr;
    int* colInd;
    float* val;
    caffe::caffe_dense2sparse(r, c, d_A, 50, &nnz, &val, &colInd, &rowPtr);
    for(int i = 0; i < nnz; ++i) {
        printf("%f--%d\t", val[i], colInd[i]);
    }
    return 0;
}
