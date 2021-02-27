#ifndef GEMM_H
#define GEMM_H
#include "opencl.h"

void gemm_bin(int M, int N, int K, float ALPHA, 
        char  *A, int lda, 
        float *B, int ldb,
        float *C, int ldc);
        
void gemm(int TA, int TB, int M, int N, int K, float ALPHA, 
                    float *A, int lda, 
                    float *B, int ldb,
                    float BETA,
                    float *C, int ldc);

void gemm_cpu(int TA, int TB, int M, int N, int K, float ALPHA, 
        float *A, int lda, 
        float *B, int ldb,
        float BETA,
        float *C, int ldc);

#ifdef GPU
void gemm_offset_gpu(int TA, int TB, int M, int N, int K,
              float ALPHA,
              cl_mem_ext A_gpu, int offset_A, int lda,
              cl_mem_ext B_gpu, int offset_B, int ldb,
              float BETA,
              cl_mem_ext C_gpu, int offset_C, int ldc);

void gemm_gpu(int TA, int TB, int M, int N, int K, float ALPHA,
              cl_mem_ext A, int lda,
              cl_mem_ext B, int ldb,
              float BETA,
              cl_mem_ext C, int ldc);
#endif
#endif
