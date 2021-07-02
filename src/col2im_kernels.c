#include <stdlib.h>
#include <string.h>

#include "col2im.h"
#include "opencl.h"
#include "col2im_kernels.cl"

#ifdef GPU

cl_program* opencl_col2im_program;
cl_kernel* opencl_col2im_kernel;

void col2im_kernel_init(void)
{
    if (opencl_device_id_t == 0) {
        opencl_col2im_program = calloc(opencl_device_ct_t, sizeof(cl_program));
        opencl_col2im_kernel = calloc(opencl_device_ct_t, sizeof(cl_kernel));
    }

    opencl_load_buffer(col2im_kernel_source, strlen(col2im_kernel_source), &opencl_col2im_program[opencl_device_id_t]);
    opencl_create_kernel(&opencl_col2im_program[opencl_device_id_t], "col2im_gpu_kernel", &opencl_col2im_kernel[opencl_device_id_t]);
}

void col2im_kernel_release(void)
{
    clReleaseKernel(opencl_col2im_kernel[opencl_device_id_t]); opencl_col2im_kernel[opencl_device_id_t] = 0;
    clReleaseProgram(opencl_col2im_program[opencl_device_id_t]); opencl_col2im_program[opencl_device_id_t] = 0;

    if (opencl_device_id_t == opencl_device_ct_t-1) {
        free(opencl_col2im_program);
        free(opencl_col2im_kernel);
    }
}

// src: https://github.com/BVLC/caffe/blob/master/src/caffe/util/im2col.cu
// You may also want to read: https://github.com/BVLC/caffe/blob/master/LICENSE

void col2im_gpu(cl_mem data_col, int offset,
        int channels, int height, int width,
        int ksize, int stride, int pad, cl_mem data_im){
    // We are going to launch channels * height_col * width_col kernels, each
    // kernel responsible for copying a single-channel grid.
    int height_col = (height + 2 * pad - ksize) / stride + 1;
    int width_col = (width + 2 * pad - ksize) / stride + 1;
    int num_kernels = channels * height * width;

    dim2 dimGrid;
    dimGrid = dim2_create(num_kernels, 1);

    int zero = 0;

    opencl_kernel(opencl_col2im_kernel[opencl_device_id_t], dimGrid, 24,
        &num_kernels, sizeof(cl_int),
        &data_col, sizeof(cl_mem),
        &height, sizeof(cl_int),
        &width, sizeof(cl_int),
        &ksize, sizeof(cl_int),
        &pad, sizeof(cl_int),
        &stride, sizeof(cl_int),
        &height_col, sizeof(cl_int),
        &width_col, sizeof(cl_int),
        &data_im, sizeof(cl_mem),
        &zero, sizeof(cl_int),
        &offset, sizeof(cl_int));
}

#endif //GPU
