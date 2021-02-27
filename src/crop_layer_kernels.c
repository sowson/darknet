#include "darknet.h"

#include "crop_layer.h"
#include "utils.h"
#include "opencl.h"
#include "image.h"

#include "crop_layer_kernels.cl"

#ifdef GPU

cl_program* opencl_crop_layer_program;
cl_kernel* opencl_levels_image_kernel;
cl_kernel* opencl_forward_crop_layer_kernel;

void crop_kernel_init(void)
{
    if (opencl_device_id_t == 0) {
        opencl_crop_layer_program = calloc(opencl_device_ct_t, sizeof(cl_program));
        opencl_levels_image_kernel = calloc(opencl_device_ct_t, sizeof(cl_kernel));
        opencl_forward_crop_layer_kernel = calloc(opencl_device_ct_t, sizeof(cl_kernel));
    }

    opencl_load_buffer(crop_layer_kernel_source, strlen(crop_layer_kernel_source), &opencl_crop_layer_program[opencl_device_id_t]);

    opencl_create_kernel(&opencl_crop_layer_program[opencl_device_id_t], "levels_image_kernel", &opencl_levels_image_kernel[opencl_device_id_t]);
    opencl_create_kernel(&opencl_crop_layer_program[opencl_device_id_t], "forward_crop_layer_kernel", &opencl_forward_crop_layer_kernel[opencl_device_id_t]);
}

void crop_kernel_release(void)
{
    clReleaseKernel(opencl_levels_image_kernel[opencl_device_id_t]); opencl_levels_image_kernel[opencl_device_id_t] = 0;
    clReleaseKernel(opencl_forward_crop_layer_kernel[opencl_device_id_t]); opencl_forward_crop_layer_kernel[opencl_device_id_t] = 0;

    clReleaseProgram(opencl_crop_layer_program[opencl_device_id_t]); opencl_crop_layer_program[opencl_device_id_t] = 0;

    if (opencl_device_id_t == opencl_device_ct_t-1) {
        free(opencl_crop_layer_program);
        free(opencl_levels_image_kernel);
        free(opencl_forward_crop_layer_kernel);
    }
}

void forward_crop_layer_gpu(crop_layer layer, network net)
{
    layer.rand_gpu = opencl_random(layer.rand_gpu, layer.batch*8);

    float radians = layer.angle*3.14159265/180.;

    float scale = 2;
    float translate = -1;
    if(layer.noadjust){
        scale = 1;
        translate = 0;
    }

    int size = layer.batch * layer.w * layer.h;

    dim2 dimGrid;
    dimGrid = opencl_gridsize(size);

    opencl_kernel(opencl_levels_image_kernel[opencl_device_id_t], dimGrid, 22,
      &net.input_gpu.mem, sizeof(cl_mem),
      &layer.rand_gpu.mem, sizeof(cl_mem),
      &layer.batch, sizeof(cl_int),
      &layer.w, sizeof(cl_int),
      &layer.h, sizeof(cl_int),
      &net.train, sizeof(cl_int),
      &layer.saturation, sizeof(cl_float),
      &layer.exposure, sizeof(cl_float),
      &translate, sizeof(cl_float),
      &scale, sizeof(cl_float),
      &layer.shift, sizeof(cl_float));

    size = layer.batch*layer.c*layer.out_w*layer.out_h;

    opencl_kernel(opencl_forward_crop_layer_kernel[opencl_device_id_t], dimGrid, 24,
      &net.input_gpu.mem, sizeof(cl_mem),
      &layer.rand_gpu.mem, sizeof(cl_mem),
      &size, sizeof(cl_int),
      &layer.c, sizeof(cl_int),
      &layer.h, sizeof(cl_int),
      &layer.w, sizeof(cl_int),
      &layer.out_h, sizeof(cl_int),
      &layer.out_w, sizeof(cl_int),
      &net.train, sizeof(cl_int),
      &layer.flip, sizeof(cl_int),
      &radians, sizeof(cl_float),
      &layer.output_gpu.mem, sizeof(cl_mem));
}

#endif