#ifdef GPU

#include "darknet.h"

#include <string.h>
#include "activation_layer.h"
#include "opencl.h"

#include "activation_kernels.cl"

cl_program* opencl_activation_kernel_program;
cl_kernel* opencl_activate_array_kernel;
cl_kernel* opencl_gradient_array_kernel;

void activation_kernel_init(void)
{
	if (opencl_device_id_t == 0) {
		opencl_activation_kernel_program = (cl_program*)calloc(opencl_device_ct_t, sizeof(cl_program));
		opencl_activate_array_kernel = (cl_kernel*)calloc(opencl_device_ct_t, sizeof(cl_kernel));
        opencl_gradient_array_kernel = (cl_kernel*)calloc(opencl_device_ct_t, sizeof(cl_kernel));
	}
	opencl_load_buffer(activation_kernels_source, strlen(activation_kernels_source), &opencl_activation_kernel_program[opencl_device_id_t]);
	opencl_create_kernel(&opencl_activation_kernel_program[opencl_device_id_t],
		"activate_array_kernel", &opencl_activate_array_kernel[opencl_device_id_t]);
    opencl_create_kernel(&opencl_activation_kernel_program[opencl_device_id_t],
        "gradient_array_kernel", &opencl_gradient_array_kernel[opencl_device_id_t]);
}

void activation_kernel_release(void)
{
    clReleaseKernel(opencl_activate_array_kernel[opencl_device_id_t]);
    clReleaseKernel(opencl_gradient_array_kernel[opencl_device_id_t]);
	clReleaseProgram(opencl_activation_kernel_program[opencl_device_id_t]);

	opencl_activate_array_kernel[opencl_device_id_t] = 0;
	opencl_gradient_array_kernel[opencl_device_id_t] = 0;
	opencl_activation_kernel_program[opencl_device_id_t] = 0;

	if (opencl_device_id_t == opencl_device_ct_t-1) {
		free(opencl_activation_kernel_program);
        free(opencl_activate_array_kernel);
        free(opencl_gradient_array_kernel);
    }
}

void activate_array_offset_gpu(cl_mem_ext x, int offset, int n, ACTIVATION a)
{
	dim2 dimN;
	dimN = opencl_gridsize(n);
	opencl_kernel(opencl_activate_array_kernel[opencl_device_id_t], dimN, 8, &x.mem, sizeof(cl_mem), &offset, sizeof(cl_int), &n, sizeof(cl_int), &a, sizeof(cl_int));
}

void activate_array_gpu(cl_mem_ext x, int n, ACTIVATION a)
{
    activate_array_offset_gpu(x, 0, n, a);
}

void gradient_array_offset_gpu(cl_mem_ext x, int offset, int n, ACTIVATION a, cl_mem_ext delta)
{
	dim2 dimN;
	dimN = opencl_gridsize(n);
	opencl_kernel(opencl_gradient_array_kernel[opencl_device_id_t], dimN, 10, &x.mem, sizeof(cl_mem), &offset, sizeof(cl_int), &n, sizeof(cl_int), &a, sizeof(cl_int), &delta.mem, sizeof(cl_mem));
}

void gradient_array_gpu(cl_mem_ext x, int n, ACTIVATION a, cl_mem_ext delta)
{
    gradient_array_offset_gpu(x, 0, n, a, delta);
}

#endif // GPU