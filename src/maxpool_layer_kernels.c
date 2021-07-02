#include <string.h>

#include "maxpool_layer.h"
#include "opencl.h"
#include "maxpool_layer_kernels.cl"

#ifdef GPU

cl_program* opencl_maxpool_layer_kernels_program;
cl_kernel* opencl_forward_maxpool_layer_kernel;
cl_kernel* opencl_backward_maxpool_layer_kernel;

void maxpool_kernel_init(void)
{
    if (opencl_device_id_t == 0) {
        opencl_maxpool_layer_kernels_program = calloc(opencl_device_ct_t, sizeof(cl_program));
        opencl_forward_maxpool_layer_kernel = calloc(opencl_device_ct_t, sizeof(cl_kernel));
        opencl_backward_maxpool_layer_kernel = calloc(opencl_device_ct_t, sizeof(cl_kernel));
    }
    opencl_load_buffer(maxpool_kernel_source, strlen(maxpool_kernel_source), &opencl_maxpool_layer_kernels_program[opencl_device_id_t]);
    opencl_create_kernel(&opencl_maxpool_layer_kernels_program[opencl_device_id_t], "forward_maxpool_layer_kernel", &opencl_forward_maxpool_layer_kernel[opencl_device_id_t]);
    opencl_create_kernel(&opencl_maxpool_layer_kernels_program[opencl_device_id_t], "backward_maxpool_layer_kernel", &opencl_backward_maxpool_layer_kernel[opencl_device_id_t]);
}

void maxpool_kernel_release(void)
{
    clReleaseKernel(opencl_forward_maxpool_layer_kernel[opencl_device_id_t]); opencl_forward_maxpool_layer_kernel[opencl_device_id_t] = 0;
    clReleaseKernel(opencl_backward_maxpool_layer_kernel[opencl_device_id_t]); opencl_backward_maxpool_layer_kernel[opencl_device_id_t] = 0;
    clReleaseProgram(opencl_maxpool_layer_kernels_program[opencl_device_id_t]); opencl_maxpool_layer_kernels_program[opencl_device_id_t] = 0;

    if (opencl_device_id_t == opencl_device_ct_t-1) {
        free(opencl_maxpool_layer_kernels_program);
        free(opencl_forward_maxpool_layer_kernel);
        free(opencl_backward_maxpool_layer_kernel);
    }
}

void forward_maxpool_layer_gpu(maxpool_layer layer, network net)
{
    int h = layer.out_h;
    int w = layer.out_w;
    int c = layer.c;

    size_t n = h*w*c*layer.batch;

    dim2 dimN;
    dimN = opencl_gridsize(n);

    opencl_kernel(opencl_forward_maxpool_layer_kernel[opencl_device_id_t], dimN, 20, &n, sizeof(cl_int), &layer.h, sizeof(cl_int), &layer.w, sizeof(cl_int), &layer.c, sizeof(cl_int), &layer.stride, sizeof(cl_int), &layer.size, sizeof(cl_int), &layer.pad, sizeof(cl_int), &net.input_gpu.mem, sizeof(cl_mem), &layer.output_gpu.mem, sizeof(cl_mem), &layer.indexes_gpu.mem, sizeof(cl_mem));
}

void backward_maxpool_layer_gpu(maxpool_layer layer, network net)
{
    size_t n = layer.h*layer.w*layer.c*layer.batch;

    dim2 dimN;
    dimN = opencl_gridsize(n);

    opencl_kernel(opencl_backward_maxpool_layer_kernel[opencl_device_id_t], dimN, 20, &n, sizeof(cl_int), &layer.h, sizeof(cl_int), &layer.w, sizeof(cl_int), &layer.c, sizeof(cl_int), &layer.stride, sizeof(cl_int), &layer.size, sizeof(cl_int), &layer.pad, sizeof(cl_int), &layer.delta_gpu.mem, sizeof(cl_mem), &net.delta_gpu.mem, sizeof(cl_mem), &layer.indexes_gpu.mem, sizeof(cl_mem));
}

#endif // GPU