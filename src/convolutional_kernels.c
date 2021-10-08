#ifdef GPU

#include "darknet.h"

#include <string.h>

#include "activations.h"
#include "convolutional_layer.h"
#include "batchnorm_layer.h"
#include "gemm.h"
#include "blas.h"
#include "im2col.h"
#include "col2im.h"
#include "opencl.h"
#include "convolutional_kernels.cl"
#include "layer.h"

cl_program* opencl_convolutional_kernels_program;
cl_kernel* opencl_binarize_kernel;
cl_kernel* opencl_binarize_input_kernel;
cl_kernel* opencl_binarize_weights_kernel;
cl_kernel* opencl_smooth_kernel;

void convolutional_kernel_init(void)
{
    if (opencl_device_id_t == 0) {
        opencl_convolutional_kernels_program = (cl_program*)calloc(opencl_device_ct_t, sizeof(cl_program));
        opencl_binarize_kernel = (cl_kernel*)calloc(opencl_device_ct_t, sizeof(cl_kernel));
        opencl_binarize_input_kernel = (cl_kernel*)calloc(opencl_device_ct_t, sizeof(cl_kernel));
        opencl_binarize_weights_kernel = (cl_kernel*)calloc(opencl_device_ct_t, sizeof(cl_kernel));
        opencl_smooth_kernel = (cl_kernel*)calloc(opencl_device_ct_t, sizeof(cl_kernel));
    }

    opencl_load_buffer(convolutional_kernel_source, strlen(convolutional_kernel_source), &opencl_convolutional_kernels_program[opencl_device_id_t]);

    opencl_create_kernel(&opencl_convolutional_kernels_program[opencl_device_id_t], "binarize_kernel", &opencl_binarize_kernel[opencl_device_id_t]);
    opencl_create_kernel(&opencl_convolutional_kernels_program[opencl_device_id_t], "binarize_input_kernel", &opencl_binarize_input_kernel[opencl_device_id_t]);
    opencl_create_kernel(&opencl_convolutional_kernels_program[opencl_device_id_t], "binarize_weights_kernel", &opencl_binarize_weights_kernel[opencl_device_id_t]);
    opencl_create_kernel(&opencl_convolutional_kernels_program[opencl_device_id_t], "smooth_kernel", &opencl_smooth_kernel[opencl_device_id_t]);

}

void convolutional_kernel_release(void)
{
    clReleaseKernel(opencl_binarize_kernel[opencl_device_id_t]); opencl_binarize_kernel[opencl_device_id_t] = 0;
    clReleaseKernel(opencl_binarize_input_kernel[opencl_device_id_t]); opencl_binarize_input_kernel[opencl_device_id_t] = 0;
    clReleaseKernel(opencl_binarize_weights_kernel[opencl_device_id_t]); opencl_binarize_weights_kernel[opencl_device_id_t] = 0;
    clReleaseKernel(opencl_smooth_kernel[opencl_device_id_t]); opencl_smooth_kernel[opencl_device_id_t] = 0;
    clReleaseProgram(opencl_convolutional_kernels_program[opencl_device_id_t]); opencl_convolutional_kernels_program[opencl_device_id_t] = 0;

    if (opencl_device_id_t == opencl_device_ct_t-1) {
        free(opencl_convolutional_kernels_program);
        free(opencl_binarize_kernel);
        free(opencl_binarize_input_kernel);
        free(opencl_binarize_weights_kernel);
        free(opencl_smooth_kernel);
    }
}

void binarize_gpu(cl_mem x, int n, cl_mem binary)
{

    dim2 dimN;
    dimN = opencl_gridsize(n);

    opencl_kernel(opencl_binarize_kernel[opencl_device_id_t], dimN, 6, &x, sizeof(cl_mem), &n, sizeof(cl_int), &binary, sizeof(cl_mem));
}


void binarize_input_gpu(cl_mem input, int n, int size, cl_mem binary)
{
    dim2 dimN;
    dimN = opencl_gridsize(size);

    opencl_kernel(opencl_binarize_input_kernel[opencl_device_id_t], dimN, 8, &input, sizeof(cl_mem), &n, sizeof(cl_int), &size, sizeof(cl_int), &binary, sizeof(cl_mem));
}


void binarize_weights_gpu(cl_mem weights, int n, int size, cl_mem binary)
{
    dim2 dimN;
    dimN = opencl_gridsize(n);

    opencl_kernel(opencl_binarize_weights_kernel[opencl_device_id_t], dimN, 8, &weights, sizeof(cl_mem), &n, sizeof(cl_int), &size, sizeof(cl_int), &binary, sizeof(cl_mem));
}

void swap_binary_gpu(convolutional_layer *l)
{
    cl_mem_ext swap_gpu = l->weights_gpu;
    l->weights_gpu = l->binary_weights_gpu;
    l->binary_weights_gpu = swap_gpu;
}

void forward_convolutional_layer_gpu(convolutional_layer l, network net)
{
    fill_gpu(l.outputs*l.batch, 0, l.output_gpu, 1);
    if(l.binary){
        binarize_weights_gpu(l.weights_gpu.mem, l.n, l.c/l.groups*l.size*l.size, l.binary_weights_gpu.mem);
        swap_binary_gpu(&l);
    }

    if(l.xnor){
        binarize_weights_gpu(l.weights_gpu.mem, l.n, l.c/l.groups*l.size*l.size, l.binary_weights_gpu.mem);
        swap_binary_gpu(&l);
        binarize_gpu(net.input_gpu.mem, l.c*l.h*l.w*l.batch, l.binary_input_gpu.mem);
        net.input_gpu = l.binary_input_gpu;
    }

    int i, j;
    int m = l.n/l.groups;
    int k = l.size*l.size*l.c/l.groups;
    int n = l.out_w*l.out_h;
    for(i = 0; i < l.batch; ++i){
        for(j = 0; j < l.groups; ++j){
            cl_mem_ext a = l.weights_gpu; // + j*l.nweights/l.groups;
            cl_mem_ext b = net.workspace_gpu;
            cl_mem_ext c = l.output_gpu; // + (i*l.groups + j)*n*m;
            cl_mem_ext im = net.input_gpu; // + (i*l.groups + j)*l.c/l.groups*l.h*l.w;

            if (l.size == 1){
                b = im;
                gemm_offset_gpu(0,0,m,n,k,1,a,j*l.nweights/l.groups,k,b,(i*l.groups + j)*l.c/l.groups*l.h*l.w,n,1,c,(i*l.groups + j)*n*m,n);
            } else {
                im2col_gpu(im.mem, (i*l.groups + j)*l.c/l.groups*l.h*l.w, l.c/l.groups, l.h, l.w, l.size, l.stride, l.pad, b.mem);
                gemm_offset_gpu(0,0,m,n,k,1,a,j*l.nweights/l.groups,k,b,0,n,1,c,(i*l.groups + j)*n*m,n);
            }
        }
    }

    if (l.batch_normalize) {
        forward_batchnorm_layer_gpu(l, net);
    } else {
        add_bias_gpu(l.output_gpu, l.biases_gpu, l.batch, l.n, l.out_w*l.out_h);
    }

    activate_array_gpu(l.output_gpu, l.outputs*l.batch, l.activation);
    //if(l.dot > 0) dot_error_gpu(l);
    if(l.binary || l.xnor) swap_binary_gpu(&l);
}

void smooth_layer(layer l, int size, float rate)
{
    int h = l.out_h;
    int w = l.out_w;
    int c = l.out_c;

    size_t n = h*w*c*l.batch;

    dim2 dimN;
    dimN = opencl_gridsize((const int) n);

    opencl_kernel(opencl_smooth_kernel[opencl_device_id_t], dimN, 16, &l.output_gpu.mem, sizeof(cl_mem), &n, sizeof(cl_int), &l.w, sizeof(cl_int), &l.h, sizeof(cl_int), &l.c, sizeof(cl_int), &size, sizeof(cl_int), &rate, sizeof(cl_float), &l.delta_gpu.mem, sizeof(cl_mem));
}

void backward_convolutional_layer_gpu(convolutional_layer l, network net)
{
    if(l.smooth){
        smooth_layer(l, 5, l.smooth);
    }

    //constrain_gpu(l.outputs*l.batch, 1, net.delta_gpu, 1);

    gradient_array_gpu(l.output_gpu, l.outputs*l.batch, l.activation, l.delta_gpu);


    if(l.batch_normalize){
        backward_batchnorm_layer_gpu(l, net);
    } else {
        backward_bias_gpu(l.bias_updates_gpu, l.delta_gpu, l.batch, l.n, l.out_w*l.out_h);
    }
    cl_mem_ext original_input = net.input_gpu;

    if(l.xnor) net.input_gpu = l.binary_input_gpu;

    int m = l.n/l.groups;
    int n = l.size*l.size*l.c/l.groups;
    int k = l.out_w*l.out_h;

    int i, j;
    for(i = 0; i < l.batch; ++i){
        for(j = 0; j < l.groups; ++j){
            cl_mem_ext a = l.delta_gpu; // + (i*l.groups + j)*m*k;
            cl_mem_ext b = net.workspace_gpu;
            cl_mem_ext c = l.weight_updates_gpu; // + j*l.nweights/l.groups;

            cl_mem_ext im  = net.input_gpu; // + (i*l.groups + j)*l.c/l.groups*l.h*l.w;
            cl_mem_ext imd = net.delta_gpu; // + (i*l.groups + j)*l.c/l.groups*l.h*l.w;

            im2col_gpu(im.mem, (i*l.groups + j)*l.c/l.groups*l.h*l.w, l.c/l.groups, l.h, l.w, l.size, l.stride, l.pad, b.mem);
            gemm_offset_gpu(0,1,m,n,k,1,a,(i*l.groups + j)*m*k,k,b,0,k,1,c,j*l.nweights/l.groups,n);

            if (net.delta_gpu.ptr) {
                if (l.binary || l.xnor) swap_binary_gpu(&l);
                a = l.weights_gpu; // + j*l.nweights/l.groups;
                b = l.delta_gpu; // + (i*l.groups + j)*m*k;
                c = net.workspace_gpu;

                if (l.size == 1) {
                    c = imd;
                    gemm_offset_gpu(1,0,n,k,m,1,a,j*l.nweights/l.groups,n,b,(i*l.groups + j)*m*k,k,0,c,(i*l.groups + j)*l.c/l.groups*l.h*l.w,k);
                }
                else {
                    gemm_offset_gpu(1,0,n,k,m,1,a,j*l.nweights/l.groups,n,b,(i*l.groups + j)*m*k,k,0,c,0,k);
                }

                if (l.size != 1) {
                    col2im_gpu(net.workspace_gpu.mem, (i*l.groups + j)*l.c/l.groups*l.h*l.w, l.c/l.groups, l.h, l.w, l.size, l.stride, l.pad, imd.mem);
                }
                if(l.binary || l.xnor) {
                    swap_binary_gpu(&l);
                }
            }
            if(l.xnor) gradient_array_offset_gpu(original_input, i*l.c*l.h*l.w, l.c*l.h*l.w, HARDTAN, net.delta_gpu);
        }
    }
}

void pull_convolutional_layer(layer l)
{
    opencl_pull_array(l.weights_gpu, l.weights, l.nweights);
    opencl_pull_array(l.biases_gpu, l.biases, l.n);
    opencl_pull_array(l.weight_updates_gpu, l.weight_updates, l.nweights);
    opencl_pull_array(l.bias_updates_gpu, l.bias_updates, l.n);
    if (l.batch_normalize){
        opencl_pull_array(l.scales_gpu, l.scales, l.n);
        opencl_pull_array(l.rolling_mean_gpu, l.rolling_mean, l.n);
        opencl_pull_array(l.rolling_variance_gpu, l.rolling_variance, l.n);
    }
}

void push_convolutional_layer(layer l)
{
    opencl_push_array(l.weights_gpu, l.weights, l.nweights);
    opencl_push_array(l.biases_gpu, l.biases, l.n);
    opencl_push_array(l.weight_updates_gpu, l.weight_updates, l.nweights);
    opencl_push_array(l.bias_updates_gpu, l.bias_updates, l.n);
    if (l.batch_normalize){
        opencl_push_array(l.scales_gpu, l.scales, l.n);
        opencl_push_array(l.rolling_mean_gpu, l.rolling_mean, l.n);
        opencl_push_array(l.rolling_variance_gpu, l.rolling_variance, l.n);
    }
}

void adam_update_gpu(cl_mem_ext w, cl_mem_ext d, cl_mem_ext m, cl_mem_ext v, float B1, float B2, float eps, float decay, float rate, int n, int batch, int t)
{
    scal_gpu(n, B1, m, 1);
    scal_gpu(n, B2, v, 1);
    axpy_gpu(n, -decay*batch, w, 1, d, 1);

    axpy_gpu(n, (1-B1), d, 1, m, 1);
    mul_gpu(n, d, 1, d, 1);
    axpy_gpu(n, (1-B2), d, 1, v, 1);

    adam_gpu(n, w, m, v, B1, B2, rate/batch, eps, t);
    fill_gpu(n, 0, d, 1);
}

void update_convolutional_layer_gpu(convolutional_layer l, update_args a)
{
    float learning_rate = a.learning_rate*l.learning_rate_scale;
    float momentum = a.momentum;
    float decay = a.decay;
    int batch = a.batch;

    if(a.adam){
        adam_update_gpu(l.weights_gpu, l.weight_updates_gpu, l.m_gpu, l.v_gpu, a.B1, a.B2, a.eps, decay, learning_rate, l.nweights, batch, a.t);
        adam_update_gpu(l.biases_gpu, l.bias_updates_gpu, l.bias_m_gpu, l.bias_v_gpu, a.B1, a.B2, a.eps, decay, learning_rate, l.n, batch, a.t);
        if(l.scales_gpu.ptr){
            adam_update_gpu(l.scales_gpu, l.scale_updates_gpu, l.scale_m_gpu, l.scale_v_gpu, a.B1, a.B2, a.eps, decay, learning_rate, l.n, batch, a.t);
        }
    }else{
        axpy_gpu(l.nweights, -decay*batch, l.weights_gpu, 1, l.weight_updates_gpu, 1);
        axpy_gpu(l.nweights, learning_rate/batch, l.weight_updates_gpu, 1, l.weights_gpu, 1);
        scal_gpu(l.nweights, momentum, l.weight_updates_gpu, 1);

        axpy_gpu(l.n, learning_rate/batch, l.bias_updates_gpu, 1, l.biases_gpu, 1);
        scal_gpu(l.n, momentum, l.bias_updates_gpu, 1);

        if(l.scales_gpu.ptr){
            axpy_gpu(l.n, learning_rate/batch, l.scale_updates_gpu, 1, l.scales_gpu, 1);
            scal_gpu(l.n, momentum, l.scale_updates_gpu, 1);
        }
    }
    if(l.clip){
        constrain_gpu(l.nweights, l.clip, l.weights_gpu, 1);
    }
}

#endif // GPU