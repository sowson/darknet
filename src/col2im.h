#ifndef COL2IM_H
#define COL2IM_H
#include "opencl.h"

void col2im_cpu(float* data_col,
        int channels, int height, int width,
        int ksize, int stride, int pad, float* data_im);

#ifdef GPU
void col2im_gpu(cl_mem data_col, int offset,
                int channels, int height, int width,
                int ksize, int stride, int pad, cl_mem data_im);
#endif
#endif
