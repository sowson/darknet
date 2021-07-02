#ifndef IM2COL_H
#define IM2COL_H
#include "opencl.h"

void im2col_cpu(float* data_im,
        int channels, int height, int width,
        int ksize, int stride, int pad, float* data_col);

#ifdef GPU

void im2col_gpu(cl_mem im, int index,
                int channels, int height, int width,
                int ksize, int stride, int pad,cl_mem data_col);

#endif
#endif
