#include <assert.h>
#include <string.h>
#include <math.h>

#include "darknet.h"

#include "blas.h"
#include "opencl.h"

#include "utils.h"

#include "blas_kernels_1.cl"
#include "blas_kernels_2.cl"
#include "blas_kernels_3.cl"

#ifdef GPU

#include "opencl.h"
#include "layer.h"

cl_program* opencl_blas_kernel_program1;
cl_program* opencl_blas_kernel_program2;
cl_program* opencl_blas_kernel_program3;

cl_kernel* opencl_test_kernel;
cl_kernel* softmax_device_kernel;
cl_kernel* opencl_scale_bias_kernel;
cl_kernel* opencl_backward_scale_kernel;
cl_kernel* opencl_add_bias_kernel;
cl_kernel* opencl_backward_bias_kernel;
cl_kernel* opencl_adam_kernel;
cl_kernel* opencl_normalize_kernel;
cl_kernel* opencl_normalize_delta_kernel;
cl_kernel* opencl_l2norm_kernel;
cl_kernel* opencl_variance_delta_kernel;
cl_kernel* opencl_accumulate_kernel;
cl_kernel* opencl_mean_delta_kernel;
cl_kernel* opencl_mean_kernel;
cl_kernel* opencl_variance_kernel;
cl_kernel* opencl_reorg_kernel;
cl_kernel* opencl_axpy_kernel;
cl_kernel* opencl_pow_kernel;
cl_kernel* opencl_const_kernel;
cl_kernel* opencl_constrain_kernel;
cl_kernel* opencl_supp_kernel;
cl_kernel* opencl_add_kernel;
cl_kernel* opencl_scal_kernel;
cl_kernel* opencl_fill_kernel;
cl_kernel* opencl_mask_kernel;
cl_kernel* opencl_copy_kernel;
cl_kernel* opencl_mul_kernel;
cl_kernel* opencl_fast_mean_kernel;
cl_kernel* opencl_fast_variance_kernel;
cl_kernel* opencl_fast_mean_delta_kernel;
cl_kernel* opencl_fast_variance_delta_kernel;
cl_kernel* opencl_flatten_kernel;
cl_kernel* opencl_shortcut_kernel;
cl_kernel* opencl_smooth_l1_kernel;
cl_kernel* opencl_softmax_x_ent_kernel;
cl_kernel* opencl_logistic_x_ent_kernel;
cl_kernel* opencl_l2_kernel;
cl_kernel* opencl_l1_kernel;
cl_kernel* opencl_wgan_kernel;
cl_kernel* opencl_inter_kernel;
cl_kernel* opencl_deinter_kernel;
cl_kernel* opencl_weighted_sum_kernel;
cl_kernel* opencl_weighted_delta_kernel;
cl_kernel* opencl_mult_add_into_kernel;
cl_kernel* opencl_softmax_tree_kernel;
cl_kernel* opencl_softmax_kernel;
cl_kernel* opencl_scale_mask_kernel;
cl_kernel* opencl_dot_kernel;
cl_kernel* opencl_upsample_kernel;
cl_kernel* opencl_gemm_kernel;
cl_kernel* opencl_mean_array_kernel;
cl_kernel* opencl_scal_add_kernel;

void blas_kernel_init(void)
{
    if (opencl_device_id_t == 0) {
        opencl_blas_kernel_program1 = (cl_program*)calloc(opencl_device_ct_t, sizeof(cl_program));
        opencl_blas_kernel_program2 = (cl_program*)calloc(opencl_device_ct_t, sizeof(cl_program));
        opencl_blas_kernel_program3 = (cl_program*)calloc(opencl_device_ct_t, sizeof(cl_program));

        opencl_test_kernel= (cl_kernel*)calloc(opencl_device_ct_t, sizeof(cl_kernel));
        softmax_device_kernel = (cl_kernel*)calloc(opencl_device_ct_t, sizeof(cl_kernel));
        opencl_scale_bias_kernel= (cl_kernel*)calloc(opencl_device_ct_t, sizeof(cl_kernel));
        opencl_backward_scale_kernel= (cl_kernel*)calloc(opencl_device_ct_t, sizeof(cl_kernel));
        opencl_add_bias_kernel= (cl_kernel*)calloc(opencl_device_ct_t, sizeof(cl_kernel));
        opencl_backward_bias_kernel= (cl_kernel*)calloc(opencl_device_ct_t, sizeof(cl_kernel));
        opencl_adam_kernel= (cl_kernel*)calloc(opencl_device_ct_t, sizeof(cl_kernel));
        opencl_normalize_kernel= (cl_kernel*)calloc(opencl_device_ct_t, sizeof(cl_kernel));
        opencl_normalize_delta_kernel= (cl_kernel*)calloc(opencl_device_ct_t, sizeof(cl_kernel));
        opencl_l2norm_kernel= (cl_kernel*)calloc(opencl_device_ct_t, sizeof(cl_kernel));
        opencl_variance_delta_kernel= (cl_kernel*)calloc(opencl_device_ct_t, sizeof(cl_kernel));
        opencl_accumulate_kernel= (cl_kernel*)calloc(opencl_device_ct_t, sizeof(cl_kernel));
        opencl_mean_delta_kernel= (cl_kernel*)calloc(opencl_device_ct_t, sizeof(cl_kernel));
        opencl_mean_kernel= (cl_kernel*)calloc(opencl_device_ct_t, sizeof(cl_kernel));
        opencl_variance_kernel= (cl_kernel*)calloc(opencl_device_ct_t, sizeof(cl_kernel));
        opencl_reorg_kernel= (cl_kernel*)calloc(opencl_device_ct_t, sizeof(cl_kernel));
        opencl_axpy_kernel= (cl_kernel*)calloc(opencl_device_ct_t, sizeof(cl_kernel));
        opencl_pow_kernel= (cl_kernel*)calloc(opencl_device_ct_t, sizeof(cl_kernel));
        opencl_const_kernel= (cl_kernel*)calloc(opencl_device_ct_t, sizeof(cl_kernel));
        opencl_constrain_kernel= (cl_kernel*)calloc(opencl_device_ct_t, sizeof(cl_kernel));
        opencl_supp_kernel= (cl_kernel*)calloc(opencl_device_ct_t, sizeof(cl_kernel));
        opencl_add_kernel= (cl_kernel*)calloc(opencl_device_ct_t, sizeof(cl_kernel));
        opencl_scal_kernel= (cl_kernel*)calloc(opencl_device_ct_t, sizeof(cl_kernel));
        opencl_fill_kernel= (cl_kernel*)calloc(opencl_device_ct_t, sizeof(cl_kernel));
        opencl_mask_kernel= (cl_kernel*)calloc(opencl_device_ct_t, sizeof(cl_kernel));
        opencl_copy_kernel= (cl_kernel*)calloc(opencl_device_ct_t, sizeof(cl_kernel));
        opencl_mul_kernel= (cl_kernel*)calloc(opencl_device_ct_t, sizeof(cl_kernel));
        opencl_fast_mean_kernel= (cl_kernel*)calloc(opencl_device_ct_t, sizeof(cl_kernel));
        opencl_fast_variance_kernel= (cl_kernel*)calloc(opencl_device_ct_t, sizeof(cl_kernel));
        opencl_fast_mean_delta_kernel= (cl_kernel*)calloc(opencl_device_ct_t, sizeof(cl_kernel));
        opencl_fast_variance_delta_kernel= (cl_kernel*)calloc(opencl_device_ct_t, sizeof(cl_kernel));
        opencl_flatten_kernel= (cl_kernel*)calloc(opencl_device_ct_t, sizeof(cl_kernel));
        opencl_shortcut_kernel= (cl_kernel*)calloc(opencl_device_ct_t, sizeof(cl_kernel));
        opencl_smooth_l1_kernel= (cl_kernel*)calloc(opencl_device_ct_t, sizeof(cl_kernel));
        opencl_softmax_x_ent_kernel= (cl_kernel*)calloc(opencl_device_ct_t, sizeof(cl_kernel));
        opencl_logistic_x_ent_kernel= (cl_kernel*)calloc(opencl_device_ct_t, sizeof(cl_kernel));
        opencl_l2_kernel= (cl_kernel*)calloc(opencl_device_ct_t, sizeof(cl_kernel));
        opencl_l1_kernel= (cl_kernel*)calloc(opencl_device_ct_t, sizeof(cl_kernel));
        opencl_wgan_kernel= (cl_kernel*)calloc(opencl_device_ct_t, sizeof(cl_kernel));
        opencl_inter_kernel= (cl_kernel*)calloc(opencl_device_ct_t, sizeof(cl_kernel));
        opencl_deinter_kernel= (cl_kernel*)calloc(opencl_device_ct_t, sizeof(cl_kernel));
        opencl_weighted_sum_kernel= (cl_kernel*)calloc(opencl_device_ct_t, sizeof(cl_kernel));
        opencl_weighted_delta_kernel= (cl_kernel*)calloc(opencl_device_ct_t, sizeof(cl_kernel));
        opencl_mult_add_into_kernel= (cl_kernel*)calloc(opencl_device_ct_t, sizeof(cl_kernel));
        opencl_softmax_tree_kernel= (cl_kernel*)calloc(opencl_device_ct_t, sizeof(cl_kernel));
        opencl_softmax_kernel= (cl_kernel*)calloc(opencl_device_ct_t, sizeof(cl_kernel));
        opencl_scale_mask_kernel= (cl_kernel*)calloc(opencl_device_ct_t, sizeof(cl_kernel));
        opencl_dot_kernel= (cl_kernel*)calloc(opencl_device_ct_t, sizeof(cl_kernel));
        opencl_upsample_kernel= (cl_kernel*)calloc(opencl_device_ct_t, sizeof(cl_kernel));
        opencl_gemm_kernel= (cl_kernel*)calloc(opencl_device_ct_t, sizeof(cl_kernel));
        opencl_mean_array_kernel= (cl_kernel*)calloc(opencl_device_ct_t, sizeof(cl_kernel));
        opencl_scal_add_kernel= (cl_kernel*)calloc(opencl_device_ct_t, sizeof(cl_kernel));
    }
    
    opencl_load_buffer(blas_kernel_source_1, strlen(blas_kernel_source_1), &opencl_blas_kernel_program1[opencl_device_id_t]);
    opencl_load_buffer(blas_kernel_source_2, strlen(blas_kernel_source_2), &opencl_blas_kernel_program2[opencl_device_id_t]);
    opencl_load_buffer(blas_kernel_source_3, strlen(blas_kernel_source_3), &opencl_blas_kernel_program3[opencl_device_id_t]);

    opencl_create_kernel(&opencl_blas_kernel_program1[opencl_device_id_t], "test_kernel", &opencl_test_kernel[opencl_device_id_t]);
    opencl_create_kernel(&opencl_blas_kernel_program1[opencl_device_id_t], "scale_bias_kernel", &opencl_scale_bias_kernel[opencl_device_id_t]);
    opencl_create_kernel(&opencl_blas_kernel_program1[opencl_device_id_t], "backward_scale_kernel", &opencl_backward_scale_kernel[opencl_device_id_t]);
    opencl_create_kernel(&opencl_blas_kernel_program1[opencl_device_id_t], "add_bias_kernel", &opencl_add_bias_kernel[opencl_device_id_t]);
    opencl_create_kernel(&opencl_blas_kernel_program1[opencl_device_id_t], "backward_bias_kernel", &opencl_backward_bias_kernel[opencl_device_id_t]);
	opencl_create_kernel(&opencl_blas_kernel_program1[opencl_device_id_t], "mean_kernel", &opencl_mean_kernel[opencl_device_id_t]);
    opencl_create_kernel(&opencl_blas_kernel_program1[opencl_device_id_t], "variance_kernel", &opencl_variance_kernel[opencl_device_id_t]);
    opencl_create_kernel(&opencl_blas_kernel_program1[opencl_device_id_t], "mean_delta_kernel", &opencl_mean_delta_kernel[opencl_device_id_t]);
    opencl_create_kernel(&opencl_blas_kernel_program1[opencl_device_id_t], "variance_delta_kernel", &opencl_variance_delta_kernel[opencl_device_id_t]);
    opencl_create_kernel(&opencl_blas_kernel_program1[opencl_device_id_t], "accumulate_kernel", &opencl_accumulate_kernel[opencl_device_id_t]);
    opencl_create_kernel(&opencl_blas_kernel_program1[opencl_device_id_t], "fast_mean_kernel", &opencl_fast_mean_kernel[opencl_device_id_t]);
    opencl_create_kernel(&opencl_blas_kernel_program1[opencl_device_id_t], "fast_variance_kernel", &opencl_fast_variance_kernel[opencl_device_id_t]);
    opencl_create_kernel(&opencl_blas_kernel_program1[opencl_device_id_t], "fast_mean_delta_kernel", &opencl_fast_mean_delta_kernel[opencl_device_id_t]);
    opencl_create_kernel(&opencl_blas_kernel_program1[opencl_device_id_t], "fast_variance_delta_kernel", &opencl_fast_variance_delta_kernel[opencl_device_id_t]);
	opencl_create_kernel(&opencl_blas_kernel_program1[opencl_device_id_t], "adam_kernel", &opencl_adam_kernel[opencl_device_id_t]);
    opencl_create_kernel(&opencl_blas_kernel_program1[opencl_device_id_t], "normalize_kernel", &opencl_normalize_kernel[opencl_device_id_t]);
    opencl_create_kernel(&opencl_blas_kernel_program1[opencl_device_id_t], "normalize_delta_kernel", &opencl_normalize_delta_kernel[opencl_device_id_t]);
    opencl_create_kernel(&opencl_blas_kernel_program1[opencl_device_id_t], "l2norm_kernel", &opencl_l2norm_kernel[opencl_device_id_t]);    
    opencl_create_kernel(&opencl_blas_kernel_program1[opencl_device_id_t], "reorg_kernel", &opencl_reorg_kernel[opencl_device_id_t]);
    opencl_create_kernel(&opencl_blas_kernel_program1[opencl_device_id_t], "axpy_kernel", &opencl_axpy_kernel[opencl_device_id_t]);
    opencl_create_kernel(&opencl_blas_kernel_program1[opencl_device_id_t], "pow_kernel", &opencl_pow_kernel[opencl_device_id_t]);
    opencl_create_kernel(&opencl_blas_kernel_program1[opencl_device_id_t], "const_kernel", &opencl_const_kernel[opencl_device_id_t]);
    opencl_create_kernel(&opencl_blas_kernel_program1[opencl_device_id_t], "constrain_kernel", &opencl_constrain_kernel[opencl_device_id_t]);
    opencl_create_kernel(&opencl_blas_kernel_program1[opencl_device_id_t], "supp_kernel", &opencl_supp_kernel[opencl_device_id_t]);
    opencl_create_kernel(&opencl_blas_kernel_program1[opencl_device_id_t], "add_kernel", &opencl_add_kernel[opencl_device_id_t]);
    opencl_create_kernel(&opencl_blas_kernel_program1[opencl_device_id_t], "scal_kernel", &opencl_scal_kernel[opencl_device_id_t]);
    opencl_create_kernel(&opencl_blas_kernel_program1[opencl_device_id_t], "fill_kernel", &opencl_fill_kernel[opencl_device_id_t]);
    opencl_create_kernel(&opencl_blas_kernel_program1[opencl_device_id_t], "mask_kernel", &opencl_mask_kernel[opencl_device_id_t]);
    opencl_create_kernel(&opencl_blas_kernel_program1[opencl_device_id_t], "copy_kernel", &opencl_copy_kernel[opencl_device_id_t]);
    opencl_create_kernel(&opencl_blas_kernel_program1[opencl_device_id_t], "mul_kernel", &opencl_mul_kernel[opencl_device_id_t]);

    opencl_create_kernel(&opencl_blas_kernel_program2[opencl_device_id_t], "flatten_kernel", &opencl_flatten_kernel[opencl_device_id_t]);
    opencl_create_kernel(&opencl_blas_kernel_program2[opencl_device_id_t], "shortcut_kernel", &opencl_shortcut_kernel[opencl_device_id_t]);
    opencl_create_kernel(&opencl_blas_kernel_program2[opencl_device_id_t], "smooth_l1_kernel", &opencl_smooth_l1_kernel[opencl_device_id_t]);
    opencl_create_kernel(&opencl_blas_kernel_program2[opencl_device_id_t], "softmax_x_ent_kernel", &opencl_softmax_x_ent_kernel[opencl_device_id_t]);
    opencl_create_kernel(&opencl_blas_kernel_program2[opencl_device_id_t], "logistic_x_ent_kernel", &opencl_logistic_x_ent_kernel[opencl_device_id_t]);
    opencl_create_kernel(&opencl_blas_kernel_program2[opencl_device_id_t], "l2_kernel", &opencl_l2_kernel[opencl_device_id_t]);
    opencl_create_kernel(&opencl_blas_kernel_program2[opencl_device_id_t], "l1_kernel", &opencl_l1_kernel[opencl_device_id_t]);
    opencl_create_kernel(&opencl_blas_kernel_program2[opencl_device_id_t], "wgan_kernel", &opencl_wgan_kernel[opencl_device_id_t]);
    opencl_create_kernel(&opencl_blas_kernel_program2[opencl_device_id_t], "weighted_sum_kernel", &opencl_weighted_sum_kernel[opencl_device_id_t]);

    opencl_create_kernel(&opencl_blas_kernel_program3[opencl_device_id_t], "deinter_kernel", &opencl_deinter_kernel[opencl_device_id_t]);
    opencl_create_kernel(&opencl_blas_kernel_program3[opencl_device_id_t], "inter_kernel", &opencl_inter_kernel[opencl_device_id_t]);
    opencl_create_kernel(&opencl_blas_kernel_program3[opencl_device_id_t], "softmax_device", &softmax_device_kernel[opencl_device_id_t]);
    opencl_create_kernel(&opencl_blas_kernel_program3[opencl_device_id_t], "weighted_delta_kernel", &opencl_weighted_delta_kernel[opencl_device_id_t]);
    opencl_create_kernel(&opencl_blas_kernel_program3[opencl_device_id_t], "mult_add_into_kernel", &opencl_mult_add_into_kernel[opencl_device_id_t]);
    opencl_create_kernel(&opencl_blas_kernel_program3[opencl_device_id_t], "softmax_tree_kernel", &opencl_softmax_tree_kernel[opencl_device_id_t]);
    opencl_create_kernel(&opencl_blas_kernel_program3[opencl_device_id_t], "softmax_kernel", &opencl_softmax_kernel[opencl_device_id_t]);
    opencl_create_kernel(&opencl_blas_kernel_program3[opencl_device_id_t], "scale_mask_kernel", &opencl_scale_mask_kernel[opencl_device_id_t]);
    opencl_create_kernel(&opencl_blas_kernel_program3[opencl_device_id_t], "dot_kernel", &opencl_dot_kernel[opencl_device_id_t]);
    opencl_create_kernel(&opencl_blas_kernel_program3[opencl_device_id_t], "upsample_kernel", &opencl_upsample_kernel[opencl_device_id_t]);
    opencl_create_kernel(&opencl_blas_kernel_program3[opencl_device_id_t], "gemm_kernel", &opencl_gemm_kernel[opencl_device_id_t]);
    opencl_create_kernel(&opencl_blas_kernel_program3[opencl_device_id_t], "mean_array_kernel", &opencl_mean_array_kernel[opencl_device_id_t]);
    opencl_create_kernel(&opencl_blas_kernel_program3[opencl_device_id_t], "scal_add_kernel", &opencl_scal_add_kernel[opencl_device_id_t]);
}

void blas_kernel_release(void)
{
    clReleaseKernel(opencl_test_kernel[opencl_device_id_t]); opencl_test_kernel[opencl_device_id_t] = 0;
    clReleaseKernel(softmax_device_kernel[opencl_device_id_t]); softmax_device_kernel[opencl_device_id_t] = 0;
    clReleaseKernel(opencl_scale_bias_kernel[opencl_device_id_t]); opencl_scale_bias_kernel[opencl_device_id_t] = 0;
    clReleaseKernel(opencl_backward_scale_kernel[opencl_device_id_t]); opencl_backward_scale_kernel[opencl_device_id_t] = 0;
    clReleaseKernel(opencl_add_bias_kernel[opencl_device_id_t]); opencl_add_bias_kernel[opencl_device_id_t] = 0;
    clReleaseKernel(opencl_backward_bias_kernel[opencl_device_id_t]); opencl_backward_bias_kernel[opencl_device_id_t] = 0;
    clReleaseKernel(opencl_adam_kernel[opencl_device_id_t]); opencl_adam_kernel[opencl_device_id_t] = 0;
    clReleaseKernel(opencl_normalize_kernel[opencl_device_id_t]); opencl_normalize_kernel[opencl_device_id_t] = 0;
    clReleaseKernel(opencl_normalize_delta_kernel[opencl_device_id_t]); opencl_normalize_delta_kernel[opencl_device_id_t] = 0;
    clReleaseKernel(opencl_l2norm_kernel[opencl_device_id_t]); opencl_l2norm_kernel[opencl_device_id_t] = 0;
    clReleaseKernel(opencl_variance_delta_kernel[opencl_device_id_t]); opencl_variance_delta_kernel[opencl_device_id_t] = 0;
    clReleaseKernel(opencl_accumulate_kernel[opencl_device_id_t]); opencl_accumulate_kernel[opencl_device_id_t] = 0;
    clReleaseKernel(opencl_mean_delta_kernel[opencl_device_id_t]); opencl_mean_delta_kernel[opencl_device_id_t] = 0;
    clReleaseKernel(opencl_mean_kernel[opencl_device_id_t]); opencl_mean_kernel[opencl_device_id_t] = 0;
    clReleaseKernel(opencl_variance_kernel[opencl_device_id_t]); opencl_variance_kernel[opencl_device_id_t] = 0;
    clReleaseKernel(opencl_reorg_kernel[opencl_device_id_t]); opencl_reorg_kernel[opencl_device_id_t] = 0;
    clReleaseKernel(opencl_axpy_kernel[opencl_device_id_t]); opencl_axpy_kernel[opencl_device_id_t] = 0;
    clReleaseKernel(opencl_pow_kernel[opencl_device_id_t]); opencl_pow_kernel[opencl_device_id_t] = 0;
    clReleaseKernel(opencl_const_kernel[opencl_device_id_t]); opencl_const_kernel[opencl_device_id_t] = 0;
    clReleaseKernel(opencl_constrain_kernel[opencl_device_id_t]); opencl_constrain_kernel[opencl_device_id_t] = 0;
    clReleaseKernel(opencl_supp_kernel[opencl_device_id_t]); opencl_supp_kernel[opencl_device_id_t] = 0;
    clReleaseKernel(opencl_add_kernel[opencl_device_id_t]); opencl_add_kernel[opencl_device_id_t] = 0;
    clReleaseKernel(opencl_scal_kernel[opencl_device_id_t]); opencl_scal_kernel[opencl_device_id_t] = 0;
    clReleaseKernel(opencl_fill_kernel[opencl_device_id_t]); opencl_fill_kernel[opencl_device_id_t] = 0;
    clReleaseKernel(opencl_mask_kernel[opencl_device_id_t]); opencl_mask_kernel[opencl_device_id_t] = 0;
    clReleaseKernel(opencl_copy_kernel[opencl_device_id_t]); opencl_copy_kernel[opencl_device_id_t] = 0;
    clReleaseKernel(opencl_mul_kernel[opencl_device_id_t]); opencl_mul_kernel[opencl_device_id_t] = 0;
    clReleaseKernel(opencl_fast_mean_kernel[opencl_device_id_t]); opencl_fast_mean_kernel[opencl_device_id_t] = 0;
    clReleaseKernel(opencl_fast_variance_kernel[opencl_device_id_t]); opencl_fast_variance_kernel[opencl_device_id_t] = 0;
    clReleaseKernel(opencl_fast_mean_delta_kernel[opencl_device_id_t]); opencl_fast_mean_delta_kernel[opencl_device_id_t] = 0;
    clReleaseKernel(opencl_fast_variance_delta_kernel[opencl_device_id_t]); opencl_fast_variance_delta_kernel[opencl_device_id_t] = 0;
    clReleaseKernel(opencl_flatten_kernel[opencl_device_id_t]); opencl_flatten_kernel[opencl_device_id_t] = 0;
    clReleaseKernel(opencl_shortcut_kernel[opencl_device_id_t]); opencl_shortcut_kernel[opencl_device_id_t] = 0;
    clReleaseKernel(opencl_smooth_l1_kernel[opencl_device_id_t]); opencl_smooth_l1_kernel[opencl_device_id_t] = 0;
    clReleaseKernel(opencl_softmax_x_ent_kernel[opencl_device_id_t]); opencl_softmax_x_ent_kernel[opencl_device_id_t] = 0;
    clReleaseKernel(opencl_logistic_x_ent_kernel[opencl_device_id_t]); opencl_logistic_x_ent_kernel[opencl_device_id_t] = 0;
    clReleaseKernel(opencl_l2_kernel[opencl_device_id_t]); opencl_l2_kernel[opencl_device_id_t] = 0;
    clReleaseKernel(opencl_l1_kernel[opencl_device_id_t]); opencl_l1_kernel[opencl_device_id_t] = 0;
    clReleaseKernel(opencl_wgan_kernel[opencl_device_id_t]); opencl_wgan_kernel[opencl_device_id_t] = 0;
    clReleaseKernel(opencl_deinter_kernel[opencl_device_id_t]); opencl_deinter_kernel[opencl_device_id_t] = 0;
    clReleaseKernel(opencl_inter_kernel[opencl_device_id_t]); opencl_inter_kernel[opencl_device_id_t] = 0;
    clReleaseKernel(opencl_weighted_sum_kernel[opencl_device_id_t]); opencl_weighted_sum_kernel[opencl_device_id_t] = 0;
    clReleaseKernel(opencl_weighted_delta_kernel[opencl_device_id_t]); opencl_weighted_delta_kernel[opencl_device_id_t] = 0;
    clReleaseKernel(opencl_mult_add_into_kernel[opencl_device_id_t]); opencl_mult_add_into_kernel[opencl_device_id_t] = 0;
    clReleaseKernel(opencl_softmax_tree_kernel[opencl_device_id_t]); opencl_softmax_tree_kernel[opencl_device_id_t] = 0;
    clReleaseKernel(opencl_softmax_kernel[opencl_device_id_t]); opencl_softmax_kernel[opencl_device_id_t] = 0;
    clReleaseKernel(opencl_scale_mask_kernel[opencl_device_id_t]); opencl_scale_mask_kernel[opencl_device_id_t] = 0;
    clReleaseKernel(opencl_dot_kernel[opencl_device_id_t]); opencl_dot_kernel[opencl_device_id_t] = 0;
    clReleaseKernel(opencl_upsample_kernel[opencl_device_id_t]); opencl_upsample_kernel[opencl_device_id_t] = 0;
    clReleaseKernel(opencl_gemm_kernel[opencl_device_id_t]); opencl_gemm_kernel[opencl_device_id_t] = 0;
    clReleaseKernel(opencl_mean_array_kernel[opencl_device_id_t]); opencl_mean_array_kernel[opencl_device_id_t] = 0;
    clReleaseKernel(opencl_scal_add_kernel[opencl_device_id_t]); opencl_scal_add_kernel[opencl_device_id_t] = 0;

    clReleaseProgram(opencl_blas_kernel_program3[opencl_device_id_t]); opencl_blas_kernel_program3[opencl_device_id_t] = 0;
    clReleaseProgram(opencl_blas_kernel_program2[opencl_device_id_t]); opencl_blas_kernel_program2[opencl_device_id_t] = 0;
    clReleaseProgram(opencl_blas_kernel_program1[opencl_device_id_t]); opencl_blas_kernel_program1[opencl_device_id_t] = 0;

    if (opencl_device_id_t == opencl_device_ct_t-1) {
        free(opencl_blas_kernel_program3);
        free(opencl_blas_kernel_program2);
        free(opencl_blas_kernel_program1);
        free(softmax_device_kernel);
        free(opencl_scale_bias_kernel);
        free(opencl_backward_scale_kernel);
        free(opencl_add_bias_kernel);
        free(opencl_backward_bias_kernel);
        free(opencl_adam_kernel);
        free(opencl_normalize_kernel);
        free(opencl_normalize_delta_kernel);
        free(opencl_l2norm_kernel);
        free(opencl_variance_delta_kernel);
        free(opencl_accumulate_kernel);
        free(opencl_mean_delta_kernel);
        free(opencl_mean_kernel);
        free(opencl_variance_kernel);
        free(opencl_reorg_kernel);
        free(opencl_axpy_kernel);
        free(opencl_pow_kernel);
        free(opencl_const_kernel);
        free(opencl_constrain_kernel);
        free(opencl_supp_kernel);
        free(opencl_add_kernel);
        free(opencl_scal_kernel);
        free(opencl_fill_kernel);
        free(opencl_mask_kernel);
        free(opencl_copy_kernel);
        free(opencl_mul_kernel);
        free(opencl_fast_mean_kernel);
        free(opencl_fast_variance_kernel);
        free(opencl_fast_mean_delta_kernel);
        free(opencl_fast_variance_delta_kernel);
        free(opencl_flatten_kernel);
        free(opencl_shortcut_kernel);
        free(opencl_smooth_l1_kernel);
        free(opencl_softmax_x_ent_kernel);
        free(opencl_logistic_x_ent_kernel);
        free(opencl_l2_kernel);
        free(opencl_l1_kernel);
        free(opencl_wgan_kernel);
        free(opencl_inter_kernel);
        free(opencl_deinter_kernel);
        free(opencl_weighted_sum_kernel);
        free(opencl_weighted_delta_kernel);
        free(opencl_mult_add_into_kernel);
        free(opencl_softmax_tree_kernel);
        free(opencl_softmax_kernel);
        free(opencl_scale_mask_kernel);
        free(opencl_dot_kernel);
        free(opencl_upsample_kernel);
        free(opencl_gemm_kernel);
        free(opencl_mean_array_kernel);
        free(opencl_scal_add_kernel);
        free(opencl_test_kernel);
    }
}

void test_kernel_gpu(int N, cl_mem_ext input, cl_mem_ext output, cl_mem_ext expected)
{
    dim2 dimGrid;
    dimGrid = dim2_create(1, 1);

    opencl_kernel(opencl_test_kernel[opencl_device_id_t], dimGrid, 8,
                  &N, sizeof(cl_int),
                  &input.mem, sizeof(cl_mem),
                  &output.mem, sizeof(cl_mem),
                  &expected.mem, sizeof(cl_mem)
    );
}

void scale_bias_gpu(cl_mem_ext output, cl_mem_ext biases, int batch, int n, int size)
{
    int N = batch*n*size;
    dim2 dimGrid;
    dimGrid = dim2_create(N, 1);

    opencl_kernel(opencl_scale_bias_kernel[opencl_device_id_t], dimGrid, 12, &N, sizeof(cl_int), &output.mem, sizeof(cl_mem), &biases.mem, sizeof(cl_mem), &batch, sizeof(cl_int), &n, sizeof(cl_int), &size, sizeof(cl_int));
}


void backward_scale_gpu(cl_mem_ext x_norm, cl_mem_ext delta, int batch, int n, int size, cl_mem_ext scale_updates)
{
    int tuning = 32;
    dim2 dimGridG1;
    dimGridG1 = dim2_create(tuning, n);
    dim2 dimGridL1;
    dimGridL1 = dim2_create(tuning, 1);

    opencl_kernel_local(opencl_backward_scale_kernel[opencl_device_id_t], dimGridG1, dimGridL1, 16, &tuning, sizeof(cl_int), NULL, tuning*sizeof(cl_float), &batch, sizeof(cl_int), &n, sizeof(cl_int), &size, sizeof(cl_int), &x_norm.mem, sizeof(cl_mem), &delta.mem, sizeof(cl_mem), &scale_updates.mem, sizeof(cl_mem));
}

void add_bias_gpu(cl_mem_ext output, cl_mem_ext biases, int batch, int n, int size)
{
    int N = batch*n*size;
    dim2 dimGrid;
    dimGrid = dim2_create(N, 1);

    opencl_kernel(opencl_add_bias_kernel[opencl_device_id_t], dimGrid, 12, &N, sizeof(cl_int), &output.mem, sizeof(cl_mem), &biases.mem, sizeof(cl_mem), &batch, sizeof(cl_int), &n, sizeof(cl_int), &size, sizeof(cl_int));
}


void backward_bias_gpu(cl_mem_ext bias_updates, cl_mem_ext delta, int batch, int n, int size)
{
    int tuning = 32;
    dim2 dimGridG1;
    dimGridG1 = dim2_create(tuning, n);
    dim2 dimGridL1;
    dimGridL1 = dim2_create(tuning, 1);

    opencl_kernel_local(opencl_backward_bias_kernel[opencl_device_id_t], dimGridG1, dimGridL1, 14, &tuning, sizeof(cl_int), NULL, tuning*sizeof(cl_float), &batch, sizeof(cl_int), &n, sizeof(cl_int), &size, sizeof(cl_int), &bias_updates.mem, sizeof(cl_mem), &delta.mem, sizeof(cl_mem));
}

void adam_gpu(int n, cl_mem_ext x, cl_mem_ext m, cl_mem_ext v, float B1, float B2, float rate, float eps, int t)
{
    dim2 dimGrid;
    dimGrid = opencl_gridsize(n);

    opencl_kernel(opencl_adam_kernel[opencl_device_id_t], dimGrid, 18, &n, sizeof(cl_int), &x.mem, sizeof(cl_mem), &m.mem, sizeof(cl_mem), &v.mem, sizeof(cl_mem), &B1, sizeof(cl_float), &B2, sizeof(cl_float), &rate, sizeof(cl_float), &eps, sizeof(cl_float), &t, sizeof(cl_int));
}


void normalize_gpu(cl_mem_ext x, cl_mem_ext mean, cl_mem_ext variance, int batch, int filters, int spatial)
{
    size_t N = batch*filters*spatial;
    dim2 dimN;
    dimN = opencl_gridsize(N);

    opencl_kernel(opencl_normalize_kernel[opencl_device_id_t], dimN, 14, &N, sizeof(cl_int), &x.mem, sizeof(cl_mem), &mean.mem, sizeof(cl_mem), &variance.mem, sizeof(cl_mem), &batch, sizeof(cl_int), &filters, sizeof(cl_int), &spatial, sizeof(cl_int));
}


void normalize_delta_gpu(cl_mem_ext x, cl_mem_ext mean, cl_mem_ext variance, cl_mem_ext mean_delta, cl_mem_ext variance_delta, int batch, int filters, int spatial, cl_mem_ext delta)
{
    size_t N = batch*filters*spatial;
    dim2 dimGrid;
    dimGrid = opencl_gridsize(N);

    opencl_kernel(opencl_normalize_delta_kernel[opencl_device_id_t], dimGrid, 20, &N, sizeof(cl_int), &x.mem, sizeof(cl_mem), &mean.mem, sizeof(cl_mem), &variance.mem, sizeof(cl_mem), &mean_delta.mem, sizeof(cl_mem), &variance_delta.mem, sizeof(cl_mem), &batch, sizeof(cl_int), &filters, sizeof(cl_int), &spatial, sizeof(cl_int), &delta.mem, sizeof(cl_mem));
}


void mean_gpu(cl_mem_ext x, int batch, int filters, int spatial, cl_mem_ext mean)
{
    size_t N = filters;
    dim2 dimGrid;
    dimGrid = dim2_create(N, 1);

    opencl_kernel(opencl_mean_kernel[opencl_device_id_t], dimGrid, 12, &N, sizeof(cl_int), &x.mem, sizeof(cl_mem), &batch, sizeof(cl_int), &filters, sizeof(cl_int), &spatial, sizeof(cl_int), &mean.mem, sizeof(cl_mem));
}

void variance_gpu(cl_mem_ext x, cl_mem_ext mean, int batch, int filters, int spatial, cl_mem_ext variance)
{
    size_t N = filters;
    dim2 dimGrid;
    dimGrid = dim2_create(N, 1);

    opencl_kernel(opencl_variance_kernel[opencl_device_id_t], dimGrid, 14, &N, sizeof(cl_int), &x.mem, sizeof(cl_mem), &mean.mem, sizeof(cl_mem), &batch, sizeof(cl_int), &filters, sizeof(cl_int), &spatial, sizeof(cl_int), &variance.mem, sizeof(cl_mem));
}

void mean_delta_gpu(cl_mem_ext delta, cl_mem_ext variance, int batch, int filters, int spatial, cl_mem_ext mean_delta)
{
    size_t N = filters;
    dim2 dimGrid;
    dimGrid = dim2_create(N, 1);

    opencl_kernel(opencl_mean_delta_kernel[opencl_device_id_t], dimGrid, 14, &N, sizeof(cl_int), &delta.mem, sizeof(cl_mem), &variance.mem, sizeof(cl_mem), &batch, sizeof(cl_int), &filters, sizeof(cl_int), &spatial, sizeof(cl_int), &mean_delta.mem, sizeof(cl_mem));
}

void variance_delta_gpu(cl_mem_ext x, cl_mem_ext delta, cl_mem_ext mean, cl_mem_ext variance, int batch, int filters, int spatial, cl_mem_ext variance_delta)
{
    size_t N = filters;
    dim2 dimGrid;
    dimGrid = dim2_create(N, 1);

    opencl_kernel(opencl_variance_delta_kernel[opencl_device_id_t], dimGrid, 18, &N, sizeof(cl_int), &x.mem, sizeof(cl_mem), &delta.mem, sizeof(cl_mem), &mean.mem, sizeof(cl_mem), &variance.mem, sizeof(cl_mem), &batch, sizeof(cl_int), &filters, sizeof(cl_int), &spatial, sizeof(cl_int), &variance_delta.mem, sizeof(cl_mem));
}


void l2normalize_gpu(cl_mem_ext x, cl_mem_ext dx, int batch, int filters, int spatial)
{
    size_t N = batch*spatial;
    dim2 dimN;
    dimN = opencl_gridsize(N);

    opencl_kernel(opencl_l2norm_kernel[opencl_device_id_t], dimN, 12, &N, sizeof(cl_int), &x.mem, sizeof(cl_mem), &dx.mem, sizeof(cl_mem), &batch, sizeof(cl_int), &filters, sizeof(cl_int), &spatial, sizeof(cl_int));
}

void fast_mean_gpu(cl_mem_ext x, int batch, int filters, int spatial, cl_mem_ext mean)
{
    int tuning = 32;
    dim2 dimGridG1;
    dimGridG1 = dim2_create(tuning, filters);
    dim2 dimGridL1;
    dimGridL1 = dim2_create(tuning, 1);

    opencl_kernel_local(opencl_fast_mean_kernel[opencl_device_id_t], dimGridG1, dimGridL1, 14, &tuning, sizeof(cl_int), NULL, tuning*sizeof(cl_float), &filters, sizeof(cl_int), &batch, sizeof(cl_int), &spatial, sizeof(cl_int), &x.mem, sizeof(cl_mem), &mean.mem, sizeof(cl_mem));
}

void fast_variance_gpu(cl_mem_ext x, cl_mem_ext mean, int batch, int filters, int spatial, cl_mem_ext variance)
{
    int tuning = 32;
    dim2 dimGridG1;
    dimGridG1 = dim2_create(tuning, filters);
    dim2 dimGridL1;
    dimGridL1 = dim2_create(tuning, 1);

    opencl_kernel_local(opencl_fast_variance_kernel[opencl_device_id_t], dimGridG1, dimGridL1, 16, &tuning, sizeof(cl_int), NULL, tuning*sizeof(cl_float), &filters, sizeof(cl_int), &batch, sizeof(cl_int), &spatial, sizeof(cl_int), &x.mem, sizeof(cl_mem), &mean.mem, sizeof(cl_mem), &variance.mem, sizeof(cl_mem));
}

void fast_mean_delta_gpu(cl_mem_ext delta, cl_mem_ext variance, int batch, int filters, int spatial, cl_mem_ext mean_delta)
{
    int tuning = 32;
    dim2 dimGridG1;
    dimGridG1 = dim2_create(tuning, filters);
    dim2 dimGridL1;
    dimGridL1 = dim2_create(tuning, 1);

    opencl_kernel_local(opencl_fast_mean_delta_kernel[opencl_device_id_t], dimGridG1, dimGridL1, 16, &tuning, sizeof(cl_int), NULL, tuning*sizeof(cl_float), &filters, sizeof(cl_int), &batch, sizeof(cl_int), &spatial, sizeof(cl_int), &variance.mem, sizeof(cl_mem), &delta.mem, sizeof(cl_mem), &mean_delta.mem, sizeof(cl_mem));
}

void fast_variance_delta_gpu(cl_mem_ext x, cl_mem_ext delta, cl_mem_ext mean, cl_mem_ext variance, int batch, int filters, int spatial, cl_mem_ext variance_delta)
{
    int tuning = 32;
    dim2 dimGridG1;
    dimGridG1 = dim2_create(tuning, filters);
    dim2 dimGridL1;
    dimGridL1 = dim2_create(tuning, 1);

    opencl_kernel_local(opencl_fast_variance_delta_kernel[opencl_device_id_t], dimGridG1, dimGridL1, 20, &tuning, sizeof(cl_int), NULL, tuning*sizeof(cl_float), &filters, sizeof(cl_int), &batch, sizeof(cl_int), &spatial, sizeof(cl_int), &x.mem, sizeof(cl_mem), &variance.mem, sizeof(cl_mem), &delta.mem, sizeof(cl_mem), &mean.mem, sizeof(cl_mem), &variance_delta.mem, sizeof(cl_mem));
}

void axpy_offset_gpu(int N, float ALPHA, cl_mem_ext  X, int OFFX, int INCX, cl_mem_ext  Y, int OFFY, int INCY)
{
    dim2 dimGrid;
    dimGrid = opencl_gridsize(N);

    opencl_kernel(opencl_axpy_kernel[opencl_device_id_t], dimGrid, 16, &N, sizeof(cl_int), &ALPHA, sizeof(cl_float), &X.mem, sizeof(cl_mem), &OFFX, sizeof(cl_int), &INCX, sizeof(cl_int), &Y.mem, sizeof(cl_mem), &OFFY, sizeof(cl_int), &INCY, sizeof(cl_int));
}

void axpy_gpu(int N, float ALPHA, cl_mem_ext X, int INCX, cl_mem_ext Y, int INCY)
{
    assert(N <= X.len && N <= Y.len && X.len <= Y.len);
    axpy_offset_gpu(N, ALPHA, X, 0, INCX, Y, 0, INCY);
}

void pow_offset_gpu(int N, float ALPHA, cl_mem_ext X, int OFFX, int INCX, cl_mem_ext Y, int OFFY, int INCY)
{
    dim2 dimGrid;
    dimGrid = opencl_gridsize(N);

    opencl_kernel(opencl_pow_kernel[opencl_device_id_t], dimGrid, 16, &N, sizeof(cl_int), &ALPHA, sizeof(cl_float), &X.mem, sizeof(cl_mem), &OFFX, sizeof(cl_int), &INCX, sizeof(cl_int), &Y.mem, sizeof(cl_mem), &OFFY, sizeof(cl_int), &INCY, sizeof(cl_int));
}

void pow_gpu(int N, float ALPHA, cl_mem_ext X, int INCX, cl_mem_ext Y, int INCY)
{
    assert(N <= X.len && N <= Y.len && X.len <= Y.len);
    pow_offset_gpu(N, ALPHA, X, 0, INCX, Y, 0, INCY);
}

void copy_offset_gpu(int N, cl_mem_ext X, int OFFX, int INCX, cl_mem_ext Y, int OFFY, int INCY)
{
    dim2 dimGrid;
    dimGrid = opencl_gridsize(N);

    opencl_kernel(opencl_copy_kernel[opencl_device_id_t], dimGrid, 14, &N, sizeof(cl_int), &X.mem, sizeof(cl_mem), &OFFX, sizeof(cl_int), &INCX, sizeof(cl_int), &Y.mem, sizeof(cl_mem), &OFFY, sizeof(cl_int), &INCY, sizeof(cl_int));
}

void copy_gpu(int N, cl_mem_ext X, int INCX, cl_mem_ext Y, int INCY)
{
    assert(N <= X.len && N <= Y.len && X.len <= Y.len);
    copy_offset_gpu(N, X, 0, INCX, Y, 0, INCY);
}

void mul_gpu(int N, cl_mem_ext X, int INCX, cl_mem_ext Y, int INCY)
{
    assert(N <= X.len && N <= Y.len && X.len <= Y.len);
    dim2 dimGrid;
    dimGrid = opencl_gridsize(N);

    opencl_kernel(opencl_mul_kernel[opencl_device_id_t], dimGrid, 10, &N, sizeof(cl_int), &X.mem, sizeof(cl_mem), &INCX, sizeof(cl_int), &Y.mem, sizeof(cl_mem), &INCY, sizeof(cl_int));
}

void flatten_gpu(cl_mem_ext x, int spatial, int layers, int batch, int forward, cl_mem_ext out)
{
    int size = spatial*batch*layers;
    dim2 dimGrid;
    dimGrid = opencl_gridsize(size);

    opencl_kernel(opencl_flatten_kernel[opencl_device_id_t], dimGrid, 14, &size, sizeof(cl_int), &x.mem, sizeof(cl_mem), &spatial, sizeof(cl_int), &layers, sizeof(cl_int), &batch, sizeof(cl_int), &forward, sizeof(cl_int), &out.mem, sizeof(cl_mem));
}


void reorg_gpu(cl_mem_ext x, int w, int h, int c, int batch, int stride, int forward, cl_mem_ext out)
{
    int size = w*h*c*batch;
    dim2 dimGrid;
    dimGrid = opencl_gridsize(size);

    opencl_kernel(opencl_reorg_kernel[opencl_device_id_t], dimGrid, 18, &size, sizeof(cl_int), &x.mem, sizeof(cl_mem), &w, sizeof(cl_int), &h, sizeof(cl_int), &c, sizeof(cl_int), &batch, sizeof(cl_int), &stride, sizeof(cl_int), &forward, sizeof(cl_int), &out.mem, sizeof(cl_mem));
}

void mask_gpu(int N, cl_mem_ext X, float mask_num, cl_mem_ext mask, float val)
{
    dim2 dimGrid;
    dimGrid = opencl_gridsize(N);

    opencl_kernel(opencl_mask_kernel[opencl_device_id_t], dimGrid, 10, &N, sizeof(cl_int), &X.mem, sizeof(cl_mem), &mask_num, sizeof(cl_int), &mask.mem, sizeof(cl_mem), &val, sizeof(float));
}

void const_offset_gpu(int N, float ALPHA, cl_mem_ext X, int OFFX, int INCX)
{
    dim2 dimGrid;
    dimGrid = opencl_gridsize(N);

    opencl_kernel(opencl_const_kernel[opencl_device_id_t], dimGrid, 10, &N, sizeof(cl_int), &ALPHA, sizeof(cl_float), &X.mem, sizeof(cl_mem), &OFFX, sizeof(cl_int), &INCX, sizeof(cl_int));
}


void const_gpu(int N, float ALPHA, cl_mem_ext X, int INCX)
{
    const_offset_gpu(N, ALPHA, X, 0, INCX);
}


void constrain_gpu(int N, float ALPHA, cl_mem_ext X, int INCX)
{
    dim2 dimGrid;
    dimGrid = opencl_gridsize(N);

    opencl_kernel(opencl_constrain_kernel[opencl_device_id_t], dimGrid, 8, &N, sizeof(cl_int), &ALPHA, sizeof(cl_float), &X.mem, sizeof(cl_mem), &INCX, sizeof(cl_int));
}


void add_gpu(int N, float ALPHA, cl_mem_ext X, int INCX)
{
    dim2 dimGrid;
    dimGrid = opencl_gridsize(N);

    opencl_kernel(opencl_add_kernel[opencl_device_id_t], dimGrid, 8, &N, sizeof(cl_int), &ALPHA, sizeof(cl_float), &X.mem, sizeof(cl_mem), &INCX, sizeof(cl_int));
}


void scal_gpu(int N, float ALPHA, cl_mem_ext X, int INCX)
{
    dim2 dimGrid;
    dimGrid = opencl_gridsize(N);

    opencl_kernel(opencl_scal_kernel[opencl_device_id_t], dimGrid, 8, &N, sizeof(cl_int), &ALPHA, sizeof(cl_float), &X.mem, sizeof(cl_mem), &INCX, sizeof(cl_int));
}


void supp_gpu(int N, float ALPHA, cl_mem_ext X, int INCX)
{
    dim2 dimGrid;
    dimGrid = opencl_gridsize(N);

    opencl_kernel(opencl_supp_kernel[opencl_device_id_t], dimGrid, 8, &N, sizeof(cl_int), &ALPHA, sizeof(cl_float), &X.mem, sizeof(cl_mem), &INCX, sizeof(cl_int));
}


void fill_offset_gpu(int N, float ALPHA, cl_mem_ext X, int OFFX, int INCX)
{
    dim2 dimGrid;
    dimGrid = opencl_gridsize(N);

    opencl_kernel(opencl_fill_kernel[opencl_device_id_t], dimGrid, 10, &N, sizeof(cl_int), &ALPHA, sizeof(cl_float), &X.mem, sizeof(cl_mem), &OFFX, sizeof(cl_int), &INCX, sizeof(cl_int));
/*
#ifdef BENCHMARK
    clock_t t;
    t = clock();
#endif
    clEnqueueFillBuffer(opencl_queues[opencl_device_id_t], X.mem, &ALPHA, sizeof(cl_float), OFFX*sizeof(cl_float), N*sizeof(cl_float),0, NULL, NULL);
#ifdef BENCHMARK
    t = clock() - t;
    double time_taken = ((double)t);
    printf("%s\t%d\n", "fill_kernel", (int)time_taken);
#endif
*/
}

void fill_gpu(int N, float ALPHA, cl_mem_ext X, int INCX)
{
    fill_offset_gpu(N, ALPHA, X, 0, INCX);
}

void shortcut_gpu(int batch, int w1, int h1, int c1, cl_mem_ext add, int w2, int h2, int c2, float s1, float s2, cl_mem_ext out)
{
    int minw = (w1 < w2) ? w1 : w2;
    int minh = (h1 < h2) ? h1 : h2;
    int minc = (c1 < c2) ? c1 : c2;

    int stride = w1/w2;
    int sample = w2/w1;
    assert(stride == h1/h2);
    assert(sample == h2/h1);
    if(stride < 1) stride = 1;
    if(sample < 1) sample = 1;

    int size = batch * minw * minh * minc;
    dim2 dimGrid;
    dimGrid = opencl_gridsize(size);
    opencl_kernel(opencl_shortcut_kernel[opencl_device_id_t], dimGrid, 34, &size, sizeof(cl_int), &minw, sizeof(cl_int), &minh, sizeof(cl_int), &minc, sizeof(cl_int), &stride, sizeof(cl_int), &sample, sizeof(cl_int), &batch, sizeof(cl_int), &w1, sizeof(cl_int), &h1, sizeof(cl_int), &c1, sizeof(cl_int), &add.mem, sizeof(cl_mem), &w2, sizeof(cl_int), &h2, sizeof(cl_int), &c2, sizeof(cl_int), &s1, sizeof(cl_float), &s2, sizeof(cl_float), &out.mem, sizeof(cl_mem));
}


void smooth_l1_gpu(int n, cl_mem_ext pred, cl_mem_ext truth, cl_mem_ext delta, cl_mem_ext error)
{
    dim2 dimN;
    dimN = opencl_gridsize(n);
    opencl_kernel(opencl_smooth_l1_kernel[opencl_device_id_t], dimN, 10, &n, sizeof(cl_int), &pred.mem, sizeof(cl_mem), &truth.mem, sizeof(cl_mem), &delta.mem, sizeof(cl_mem), &error.mem, sizeof(cl_mem));
}

void softmax_x_ent_gpu(int n, cl_mem_ext pred, cl_mem_ext truth, cl_mem_ext delta, cl_mem_ext error)
{
    dim2 dimN;
    dimN = opencl_gridsize(n);
    opencl_kernel(opencl_softmax_x_ent_kernel[opencl_device_id_t], dimN, 10, &n, sizeof(cl_int), &pred.mem, sizeof(cl_mem), &truth.mem, sizeof(cl_mem), &delta.mem, sizeof(cl_mem), &error.mem, sizeof(cl_mem));
}

void logistic_x_ent_gpu(int n, cl_mem_ext pred, cl_mem_ext truth, cl_mem_ext delta, cl_mem_ext error)
{
    dim2 dimN;
    dimN = opencl_gridsize(n);
    opencl_kernel(opencl_logistic_x_ent_kernel[opencl_device_id_t], dimN, 10, &n, sizeof(cl_int), &pred.mem, sizeof(cl_mem), &truth.mem, sizeof(cl_mem), &delta.mem, sizeof(cl_mem), &error.mem, sizeof(cl_mem));
}

void l2_gpu(int n, cl_mem_ext pred, cl_mem_ext truth, cl_mem_ext delta, cl_mem_ext error)
{
    dim2 dimN;
    dimN = opencl_gridsize(n);
    opencl_kernel(opencl_l2_kernel[opencl_device_id_t], dimN, 10, &n, sizeof(cl_int), &pred.mem, sizeof(cl_mem), &truth.mem, sizeof(cl_mem), &delta.mem, sizeof(cl_mem), &error.mem, sizeof(cl_mem));
}


void l1_gpu(int n, cl_mem_ext pred, cl_mem_ext truth, cl_mem_ext delta, cl_mem_ext error)
{
    dim2 dimN;
    dimN = opencl_gridsize(n);
    opencl_kernel(opencl_l1_kernel[opencl_device_id_t], dimN, 10, &n, sizeof(cl_int), &pred.mem, sizeof(cl_mem), &truth.mem, sizeof(cl_mem), &delta.mem, sizeof(cl_mem), &error.mem, sizeof(cl_mem));
}


void wgan_gpu(int n, cl_mem_ext pred, cl_mem_ext truth, cl_mem_ext delta, cl_mem_ext error)
{
    dim2 dimN;
    dimN = opencl_gridsize(n);
    opencl_kernel(opencl_wgan_kernel[opencl_device_id_t], dimN, 10, &n, sizeof(cl_int), &pred.mem, sizeof(cl_mem), &truth.mem, sizeof(cl_mem), &delta.mem, sizeof(cl_mem), &error.mem, sizeof(cl_mem));
}

void deinter_gpu(int NX, cl_mem_ext X, int NY, cl_mem_ext Y, int B, cl_mem_ext OUT)
{
    dim2 dimN;
    dimN = opencl_gridsize((NX+NY)*B);

    opencl_kernel(opencl_deinter_kernel[opencl_device_id_t], dimN, 12, &NX, sizeof(cl_int), &X.mem, sizeof(cl_mem), &NY, &Y.mem, sizeof(cl_mem), &B, sizeof(cl_int), &OUT.mem, sizeof(cl_mem));
}

void inter_gpu(int NX, cl_mem_ext X, int NY, cl_mem_ext Y, int B, cl_mem_ext OUT)
{
    dim2 dimN;
    dimN = opencl_gridsize((NX+NY)*B);

    opencl_kernel(opencl_inter_kernel[opencl_device_id_t], dimN, 12, &NX, sizeof(cl_int), &X.mem, sizeof(cl_mem), &NY, &Y.mem, sizeof(cl_mem), &B, sizeof(cl_int), &OUT.mem, sizeof(cl_mem));
}

void weighted_sum_gpu(cl_mem_ext a, cl_mem_ext b, cl_mem_ext s, int num, cl_mem_ext c)
{
    dim2 dimNum;
    dimNum = opencl_gridsize(num);

    opencl_kernel(opencl_weighted_sum_kernel[opencl_device_id_t], dimNum, 10, &num, sizeof(cl_int), &a.mem, sizeof(cl_mem), &b.mem, sizeof(cl_mem), &s.mem, sizeof(cl_mem), &c.mem, sizeof(cl_mem));
}


void weighted_delta_gpu(cl_mem_ext a, cl_mem_ext b, cl_mem_ext s, cl_mem_ext da, cl_mem_ext db, cl_mem_ext ds, int num, cl_mem_ext dc)
{
    dim2 dimNum;
    dimNum = opencl_gridsize(num);

    opencl_kernel(opencl_weighted_delta_kernel[opencl_device_id_t], dimNum, 16, &num, sizeof(cl_int), &a.mem, sizeof(cl_mem), &b.mem, sizeof(cl_mem), &s.mem, sizeof(cl_mem), &da.mem, sizeof(cl_mem), &db.mem, sizeof(cl_mem), &ds.mem, sizeof(cl_mem), &dc.mem, sizeof(cl_mem));
}


void mult_add_into_gpu(int num, cl_mem_ext a, cl_mem_ext b, cl_mem_ext c)
{
    dim2 dimNum;
    dimNum = opencl_gridsize(num);

    opencl_kernel(opencl_mult_add_into_kernel[opencl_device_id_t], dimNum, 8, &num, sizeof(cl_int), &a.mem, sizeof(cl_mem), &b.mem, sizeof(cl_mem), &c.mem, sizeof(cl_mem));
}

void softmax_tree(cl_mem_ext input, int spatial, int batch, int stride, float temp, cl_mem_ext output, tree hier)
{
    softmax_offset_tree(input, 0, spatial, batch, stride, temp, output, hier);
}

void softmax_offset_tree(cl_mem_ext input, int offset, int spatial, int batch, int stride, float temp, cl_mem_ext output, tree hier)
{
    cl_mem_ext tree_groups_size = opencl_make_int_array(hier.group_size, hier.groups);
    cl_mem_ext tree_groups_offset = opencl_make_int_array(hier.group_offset, hier.groups);

    opencl_push_int_array(tree_groups_size, hier.group_size, hier.groups);
    opencl_push_int_array(tree_groups_offset, hier.group_offset, hier.groups);

    int num = spatial*batch*hier.groups;

    dim2 dimBatch;
    dimBatch = opencl_gridsize(num);

    opencl_kernel(opencl_softmax_tree_kernel[opencl_device_id_t], dimBatch, 20,
                  &input.mem, sizeof(cl_mem),
                  &offset, sizeof(cl_int),
                  &spatial, sizeof(cl_int),
                  &batch, sizeof(cl_int),
                  &stride, sizeof(cl_int),
                  &temp, sizeof(cl_float),
                  &output.mem, sizeof(cl_mem),
                  &hier.groups, sizeof(cl_int),
                  &tree_groups_size.mem, sizeof(cl_mem),
                  &tree_groups_offset.mem, sizeof(cl_mem)
    );

    opencl_free_gpu_only(tree_groups_size);
    opencl_free_gpu_only(tree_groups_offset);
}

void softmax_offset_gpu(cl_mem_ext input, int offset, int n, int batch, int batch_offset, int groups, int group_offset, int stride, float temp, cl_mem_ext output)
{
    dim2 dimBatch;
    dimBatch = opencl_gridsize(batch * groups);
    opencl_kernel(opencl_softmax_kernel[opencl_device_id_t], dimBatch, 20, &input.mem, sizeof(cl_mem), &offset, sizeof(cl_int), &n, sizeof(cl_int), &batch, sizeof(cl_int), &batch_offset, sizeof(cl_int), &groups, sizeof(cl_int), &group_offset, sizeof(cl_int), &stride, sizeof(cl_int), &temp, sizeof(cl_float), &output.mem, sizeof(cl_mem));
}

void softmax_gpu(cl_mem_ext input, int n, int batch, int batch_offset, int groups, int group_offset, int stride, float temp, cl_mem_ext output)
{
    softmax_offset_gpu(input, 0, n, batch, batch_offset, groups, group_offset, stride, temp, output);
}

void softmax_tree_gpu(cl_mem_ext input, int spatial, int batch, int stride, float temp, cl_mem_ext output, tree hier)
{
    dim2 dimBatch;
    dimBatch = opencl_gridsize(batch * hier.groups);

    cl_mem_ext tree_groups_size = opencl_make_int_array(hier.group_size, hier.groups);
    cl_mem_ext tree_groups_offset = opencl_make_int_array(hier.group_offset, hier.groups);

    opencl_push_int_array(tree_groups_size, hier.group_size, hier.groups);
    opencl_push_int_array(tree_groups_offset, hier.group_offset, hier.groups);

    opencl_kernel(opencl_softmax_tree_kernel[opencl_device_id_t], dimBatch, 18,
        &input.mem, sizeof(cl_mem),
        &spatial, sizeof(cl_int),
        &batch, sizeof(cl_int),
        &stride, sizeof(cl_int),
        &temp, sizeof(cl_float),
        &output.mem, sizeof(cl_mem),
        &hier.groups, sizeof(cl_int),
        &tree_groups_size.mem, sizeof(cl_mem),
        &tree_groups_offset.mem, sizeof(cl_mem)
    );

    opencl_free_gpu_only(tree_groups_size);
    opencl_free_gpu_only(tree_groups_offset);
}

void scale_mask_gpu(int N, cl_mem_ext X, float mask_num, cl_mem_ext mask, float scale)
{
    dim2 dimBatch;
    dimBatch = opencl_gridsize(N);

    opencl_kernel(opencl_scale_mask_kernel[opencl_device_id_t], dimBatch, 10,
        &N, sizeof(cl_int),
        &X.mem, sizeof(cl_mem),
        &mask_num, sizeof(cl_float),
        &mask.mem, sizeof(cl_mem),
        &scale, sizeof(cl_float)
    );
}

void dot_error_gpu(layer l)
{
    dim2 dimGrid;
    dimGrid = opencl_gridsize(l.n*l.n);

    int size = l.out_w * l.out_h;

    opencl_kernel(opencl_dot_kernel[opencl_device_id_t], dimGrid, 12,
         &l.output_gpu.mem, sizeof(cl_mem),
         &l.dot, sizeof(cl_float),
         &l.batch, sizeof(cl_int),
         &l.n, sizeof(cl_int),
         &size, sizeof(cl_int),
         &l.delta_gpu.mem, sizeof(cl_mem));
}


void upsample_gpu(cl_mem_ext in, int w, int h, int c, int batch, int stride, int forward, float scale, cl_mem_ext out)
{
    size_t size = w*h*c*batch*stride*stride;

    dim2 dimGrid;
    dimGrid = opencl_gridsize(size);

    opencl_kernel(opencl_upsample_kernel[opencl_device_id_t], dimGrid, 20,
                  &size, sizeof(cl_int),
                  &in.mem, sizeof(cl_mem),
                  &w, sizeof(cl_int),
                  &h, sizeof(cl_int),
                  &c, sizeof(cl_int),
                  &batch, sizeof(cl_int),
                  &stride, sizeof(cl_int),
                  &forward, sizeof(cl_int),
                  &scale, sizeof(cl_float),
                  &out.mem, sizeof(cl_mem));
}

void mean_array_gpu(cl_mem_ext src, int N, float alpha, cl_mem_ext avg)
{
    dim2 dimBatch;
    dimBatch = opencl_gridsize(N);

    opencl_kernel(opencl_mean_array_kernel[opencl_device_id_t], dimBatch, 8,
                  &N, sizeof(cl_int),
                  &alpha, sizeof(cl_float),
                  &src.mem, sizeof(cl_mem),
                  &avg.mem, sizeof(cl_mem)
    );
}

void scal_add_gpu(int N, float ALPHA, float BETA, cl_mem_ext X, int INCX)
{
    scal_add_offset_gpu(N, ALPHA, BETA, X, 0, INCX);
}

void scal_add_offset_gpu(int N, float ALPHA, float BETA, cl_mem_ext X, int OFFX, int INCX)
{
    dim2 dimBatch;
    dimBatch = opencl_gridsize(N);

    opencl_kernel(opencl_scal_add_kernel[opencl_device_id_t], dimBatch, 12,
                  &N, sizeof(cl_int),
                  &ALPHA, sizeof(cl_float),
                  &BETA, sizeof(cl_float),
                  &X.mem, sizeof(cl_mem),
                  &OFFX, sizeof(cl_int),
                  &INCX, sizeof(cl_int)
    );
}

#if defined(GPU_MULTI) || defined(ARM)
void gemm_offset_gpu(
        int TA, int TB,
        int M, int N, int K,
        float ALPHA,
        cl_mem_ext A_gpu, int offset_A, int lda,
        cl_mem_ext B_gpu, int offset_B, int ldb,
        float BETA,
        cl_mem_ext C_gpu, int offset_C, int ldc) {
    //printf("gemm gpu: %d %d %d %d %d %f %d %d %f %d\n",TA, TB, M, N, K, ALPHA, lda, ldb, BETA, ldc);

    int tuning = 8;
    dim3 dimGridG1;
    dimGridG1 = dim3_create(tuning, N, M);
    dim3 dimGridL1;
    dimGridL1 = dim3_create(tuning, 1, 1);

    opencl_kernel_local3(opencl_gemm_kernel[opencl_device_id_t], dimGridG1, dimGridL1, 36,
                  &tuning, sizeof(cl_int),
                  NULL, tuning*sizeof(cl_float),
                  &TA, sizeof(cl_int),
                  &TB, sizeof(cl_int),
                  &M, sizeof(cl_int),
                  &N, sizeof(cl_int),
                  &K, sizeof(cl_int),
                  &ALPHA, sizeof(cl_float),
                  &A_gpu.mem, sizeof(cl_mem),
                  &offset_A, sizeof(cl_int),
                  &lda, sizeof(cl_int),
                  &B_gpu.mem, sizeof(cl_mem),
                  &offset_B, sizeof(cl_int),
                  &ldb, sizeof(cl_int),
                  &BETA, sizeof(cl_int),
                  &C_gpu.mem, sizeof(cl_mem),
                  &offset_C, sizeof(cl_int),
                  &ldc, sizeof(cl_int)
    );
}
#endif
#endif // GPU
