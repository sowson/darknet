#ifndef __IM2COL_KERNELS_CL__
#define __IM2COL_KERNELS_CL__

// src: https://github.com/BVLC/caffe/blob/master/src/caffe/util/im2col.cu
// You may also want to read: https://github.com/BVLC/caffe/blob/master/LICENSE

static const char* const im2col_kernel_source = CONVERT_KERNEL_TO_STRING(

__kernel void im2col_gpu_kernel(int n, __global float* data_im,
                                int height, int width, int ksize,
                                int pad,
                                int stride,
                                int height_col, int width_col,
                                __global float *data_col,
                                int col_offset,
                                int im_offset) {
    int index = get_global_id(1) * get_global_size(0) + get_global_id(0);
    for(; index < n; index += get_global_size(1) * get_global_size(0)){
        int w_out = index % width_col;
        int h_index = index / width_col;
        int h_out = h_index % height_col;
        int channel_in = h_index / height_col;
        int channel_out = channel_in * ksize * ksize;
        int h_in = h_out * stride - pad;
        int w_in = w_out * stride - pad;

        int data_col_offset = (channel_out * height_col + h_out) * width_col + w_out + col_offset;
        int data_im_offset = (channel_in * height + h_in) * width + w_in + im_offset;

        for (int i = 0; i < ksize; ++i) {
            for (int j = 0; j < ksize; ++j) {
                int h = h_in + i;
                 int w = w_in + j;

                 data_col[data_col_offset] = (h >= 0 && w >= 0 && h < height && w < width) ?
                                             data_im[data_im_offset + i * width + j] : 0;

                 //data_col[data_col_offset] = data_im[data_im_offset + i * width + j];

                 data_col_offset += height_col * width_col;
            }
        }
    }
}

);

#endif
