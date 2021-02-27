#ifndef __COL2IM_KERNELS_CL__
#define __COL2IM_KERNELS_CL__

// The comment-out cuda code is from the src, and I would like to port to
// opencl kernel code as below.

// src: https://github.com/BVLC/caffe/blob/master/src/caffe/util/im2col.cu
// You may also want to read: https://github.com/BVLC/caffe/blob/master/LICENSE

static const char* const col2im_kernel_source = CONVERT_KERNEL_TO_STRING(

__kernel void col2im_gpu_kernel(int n, __global float* data_col,
                                int height, int width, int ksize,
                                int pad,
                                int stride,
                                int height_col, int width_col,
                                __global float *data_im,
                                int col_offset,
                                int img_offset) {
    int index = get_global_id(1) * get_global_size(0) + get_global_id(0);
    for(; index < n; index += get_global_size(1) * get_global_size(0)){
        float val = 0;
        int w = index % width + pad;
        int h = (index / width) % height + pad;
        int c = index / (width * height);
        // compute the start and end of the output
        int w_col_start = (w < ksize) ? 0 : (w - ksize) / stride + 1;
        int w_col_end = min(w / stride + 1, width_col);
        int h_col_start = (h < ksize) ? 0 : (h - ksize) / stride + 1;
        int h_col_end = min(h / stride + 1, height_col);
        // equivalent implementation
        int offset =
            (c * ksize * ksize + h * ksize + w) * height_col * width_col;
        int coeff_h_col = (1 - stride * ksize * height_col) * width_col;
        int coeff_w_col = (1 - stride * height_col * width_col);
        for (int h_col = h_col_start; h_col < h_col_end; ++h_col) {
            for (int w_col = w_col_start; w_col < w_col_end; ++w_col) {
                val += data_col[col_offset + offset + h_col * coeff_h_col + w_col * coeff_w_col];
            }
        }
        data_im[img_offset + index] += val;
    }
}
);
#endif