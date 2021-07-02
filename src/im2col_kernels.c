#include <stdlib.h> 
#include <string.h>

#include "im2col.h"
#include "opencl.h"
#include "im2col_kernels.cl"

#ifdef GPU

cl_program* opencl_im2col_kernels_program;
cl_kernel* opencl_im2col_gpu_kernel;

void im2col_kernel_init(void)
{
    if (opencl_device_id_t == 0) {
        opencl_im2col_kernels_program = calloc(opencl_device_ct_t, sizeof(cl_program));
        opencl_im2col_gpu_kernel = calloc(opencl_device_ct_t, sizeof(cl_kernel));
    }

    opencl_load_buffer(im2col_kernel_source, strlen(im2col_kernel_source), &opencl_im2col_kernels_program[opencl_device_id_t]);
    opencl_create_kernel(&opencl_im2col_kernels_program[opencl_device_id_t], "im2col_gpu_kernel", &opencl_im2col_gpu_kernel[opencl_device_id_t]);
}

void im2col_kernel_release(void)
{
    clReleaseKernel(opencl_im2col_gpu_kernel[opencl_device_id_t]); opencl_im2col_gpu_kernel[opencl_device_id_t] = 0;
    clReleaseProgram(opencl_im2col_kernels_program[opencl_device_id_t]); opencl_im2col_kernels_program[opencl_device_id_t] = 0;

    if (opencl_device_id_t == opencl_device_ct_t-1) {
	free(opencl_im2col_kernels_program);
        free(opencl_im2col_gpu_kernel);
    }
}

// src: https://github.com/BVLC/caffe/blob/master/src/caffe/util/im2col.cu
// You may also want to read: https://github.com/BVLC/caffe/blob/master/LICENSE

void im2col_gpu(cl_mem im, int offset,
                int channels, int height, int width,
                int ksize, int stride, int pad, cl_mem data_col){
    // We are going to launch channels * height_col * width_col kernels, each
    // kernel responsible for copying a single-channel grid.
    int height_col = (height + 2 * pad - ksize) / stride + 1;
    int width_col = (width + 2 * pad - ksize) / stride + 1;
    int num_kernels = channels * height_col * width_col;

    dim2 dimGrid;
    dimGrid = dim2_create(num_kernels, 1);

    int zero = 0;

    opencl_kernel(opencl_im2col_gpu_kernel[opencl_device_id_t], dimGrid, 24,
        &num_kernels, sizeof(cl_int),
        &im, sizeof(cl_mem),
        &height, sizeof(cl_int),
        &width, sizeof(cl_int),
        &ksize, sizeof(cl_int),
        &pad, sizeof(cl_int),
        &stride, sizeof(cl_int),
        &height_col, sizeof(cl_int),
        &width_col, sizeof(cl_int),
        &data_col, sizeof(cl_mem),
        &zero, sizeof(cl_int),
        &offset, sizeof(cl_int));
}

#endif // GPU
