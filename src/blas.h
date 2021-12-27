#ifndef BLAS_H
#define BLAS_H
#include "darknet.h"

void flatten(float *x, int size, int layers, int batch, int forward);
void pm(int M, int N, float *A);
float *random_matrix(int rows, int cols);
void time_random_matrix(int TA, int TB, int m, int k, int n);
void reorg_cpu(float *x, int w, int h, int c, int batch, int stride, int forward, float *out);

void test_blas();

void inter_cpu(int NX, float *X, int NY, float *Y, int B, float *OUT);
void deinter_cpu(int NX, float *X, int NY, float *Y, int B, float *OUT);
void mult_add_into_cpu(int N, float *X, float *Y, float *Z);

void const_cpu(int N, float ALPHA, float *X, int INCX);
void pow_cpu(int N, float ALPHA, float *X, int INCX, float *Y, int INCY);
void mul_cpu(int N, float *X, int INCX, float *Y, int INCY);

int test_gpu_blas();
void shortcut_cpu(int batch, int w1, int h1, int c1, float *add, int w2, int h2, int c2, float s1, float s2, float *out);

void mean_cpu(float *x, int batch, int filters, int spatial, float *mean);
void variance_cpu(float *x, float *mean, int batch, int filters, int spatial, float *variance);

void scale_bias(float *output, float *scales, int batch, int n, int size);
void backward_scale_cpu(float *x_norm, float *delta, int batch, int n, int size, float *scale_updates);
void mean_delta_cpu(float *delta, float *variance, int batch, int filters, int spatial, float *mean_delta);
void  variance_delta_cpu(float *x, float *delta, float *mean, float *variance, int batch, int filters, int spatial, float *variance_delta);
void normalize_delta_cpu(float *x, float *mean, float *variance, float *mean_delta, float *variance_delta, int batch, int filters, int spatial, float *delta);
void l2normalize_cpu(float *x, float *dx, int batch, int filters, int spatial);

void smooth_l1_cpu(int n, float *pred, float *truth, float *delta, float *error);
void l2_cpu(int n, float *pred, float *truth, float *delta, float *error);
void l1_cpu(int n, float *pred, float *truth, float *delta, float *error);
void logistic_x_ent_cpu(int n, float *pred, float *truth, float *delta, float *error);
void softmax_x_ent_cpu(int n, float *pred, float *truth, float *delta, float *error);
void weighted_sum_cpu(float *a, float *b, float *s, int num, float *c);
void weighted_delta_cpu(float *a, float *b, float *s, float *da, float *db, float *ds, int n, float *dc);

void softmax(float *input, int n, float temp, int stride, float *output);
void softmax_cpu(float *input, int n, int batch, int batch_offset, int groups, int group_offset, int stride, float temp, float *output);
void upsample_cpu(float *in, int w, int h, int c, int batch, int stride, int forward, float scale, float *out);

void get_embedding(float *src, int src_w, int src_h, int src_c, int embedding_size, int cur_w, int cur_h, int cur_n, int cur_b, float *dst);

#ifdef GPU
#include "opencl.h"
#include "tree.h"
void test_kernel_gpu(int N, cl_mem_ext input, cl_mem_ext output, cl_mem_ext expected);

void constrain_gpu(int N, float ALPHA, cl_mem_ext X, int INCX);

void axpy_gpu(int N, float ALPHA, cl_mem_ext X, int INCX, cl_mem_ext Y, int INCY);
void axpy_offset_gpu(int N, float ALPHA, cl_mem_ext X, int OFFX, int INCX, cl_mem_ext Y, int OFFY, int INCY);
void copy_gpu(int N, cl_mem_ext X, int INCX, cl_mem_ext Y, int INCY);
void copy_offset_gpu(int N, cl_mem_ext X, int OFFX, int INCX, cl_mem_ext Y, int OFFY, int INCY);
void add_gpu(int N, float ALPHA, cl_mem_ext X, int INCX);
void supp_gpu(int N, float ALPHA, cl_mem_ext X, int INCX);
void mask_gpu(int N, cl_mem_ext X, float ALPHA, cl_mem_ext Y, float BETA);
void scale_mask_gpu(int N, cl_mem_ext X, float mask_num, cl_mem_ext mask, float scale);
void const_gpu(int N, float ALPHA, cl_mem_ext X, int INCX);
void pow_gpu(int N, float ALPHA, cl_mem_ext X, int INCX, cl_mem_ext Y, int INCY);
void mul_gpu(int N, cl_mem_ext X, int INCX, cl_mem_ext Y, int INCY);

void normalize_gpu(cl_mem_ext x, cl_mem_ext mean, cl_mem_ext variance, int batch, int filters, int spatial);
void l2normalize_gpu(cl_mem_ext x, cl_mem_ext dx, int batch, int filters, int spatial);

void normalize_delta_gpu(cl_mem_ext x, cl_mem_ext mean, cl_mem_ext variance, cl_mem_ext mean_delta, cl_mem_ext variance_delta, int batch, int filters, int spatial, cl_mem_ext delta);

void mean_gpu(cl_mem_ext x, int batch, int filters, int spatial, cl_mem_ext mean);
void variance_gpu(cl_mem_ext x, cl_mem_ext mean, int batch, int filters, int spatial, cl_mem_ext variance);
void mean_delta_gpu(cl_mem_ext delta, cl_mem_ext variance, int batch, int filters, int spatial, cl_mem_ext mean_delta);
void variance_delta_gpu(cl_mem_ext x, cl_mem_ext delta, cl_mem_ext mean, cl_mem_ext variance, int batch, int filters, int spatial, cl_mem_ext variance_delta);

void fast_variance_gpu(cl_mem_ext x, cl_mem_ext mean, int batch, int filters, int spatial, cl_mem_ext variance);
void fast_mean_gpu(cl_mem_ext x, int batch, int filters, int spatial, cl_mem_ext mean);
void fast_mean_delta_gpu(cl_mem_ext delta, cl_mem_ext variance, int batch, int filters, int spatial, cl_mem_ext mean_delta);
void fast_variance_delta_gpu(cl_mem_ext x, cl_mem_ext delta, cl_mem_ext mean, cl_mem_ext variance, int batch, int filters, int spatial, cl_mem_ext variance_delta);

void shortcut_gpu(int batch, int w1, int h1, int c1, cl_mem_ext add, int w2, int h2, int c2, float s1, float s2, cl_mem_ext out);
void scale_bias_gpu(cl_mem_ext output, cl_mem_ext biases, int batch, int n, int size);
void backward_scale_gpu(cl_mem_ext x_norm, cl_mem_ext delta, int batch, int n, int size, cl_mem_ext scale_updates);
void add_bias_gpu(cl_mem_ext output, cl_mem_ext biases, int batch, int n, int size);
void backward_bias_gpu(cl_mem_ext bias_updates, cl_mem_ext delta, int batch, int n, int size);

void logistic_x_ent_gpu(int n, cl_mem_ext pred, cl_mem_ext truth, cl_mem_ext delta, cl_mem_ext error);
void softmax_x_ent_gpu(int n, cl_mem_ext pred, cl_mem_ext truth, cl_mem_ext delta, cl_mem_ext error);
void smooth_l1_gpu(int n, cl_mem_ext pred, cl_mem_ext truth, cl_mem_ext delta, cl_mem_ext error);
void l2_gpu(int n, cl_mem_ext pred, cl_mem_ext truth, cl_mem_ext delta, cl_mem_ext error);
void l1_gpu(int n, cl_mem_ext pred, cl_mem_ext truth, cl_mem_ext delta, cl_mem_ext error);
void wgan_gpu(int n, cl_mem_ext pred, cl_mem_ext truth, cl_mem_ext delta, cl_mem_ext error);
void weighted_delta_gpu(cl_mem_ext a, cl_mem_ext b, cl_mem_ext s, cl_mem_ext da, cl_mem_ext db, cl_mem_ext ds, int num, cl_mem_ext dc);
void weighted_sum_gpu(cl_mem_ext a, cl_mem_ext b, cl_mem_ext s, int num, cl_mem_ext c);
void mult_add_into_gpu(int num, cl_mem_ext a, cl_mem_ext b, cl_mem_ext c);
void inter_gpu(int NX, cl_mem_ext X, int NY, cl_mem_ext Y, int B, cl_mem_ext OUT);
void deinter_gpu(int NX, cl_mem_ext X, int NY, cl_mem_ext Y, int B, cl_mem_ext OUT);

void reorg_gpu(cl_mem_ext x, int w, int h, int c, int batch, int stride, int forward, cl_mem_ext out);

void softmax_gpu(cl_mem_ext input, int n, int batch, int batch_offset, int groups, int group_offset, int stride, float temp, cl_mem_ext output);
void adam_update_gpu(cl_mem_ext w, cl_mem_ext d, cl_mem_ext m, cl_mem_ext v, float B1, float B2, float eps, float decay, float rate, int n, int batch, int t);
void adam_gpu(int n, cl_mem_ext x, cl_mem_ext m, cl_mem_ext v, float B1, float B2, float rate, float eps, int t);

void flatten_gpu(cl_mem_ext x, int spatial, int layers, int batch, int forward, cl_mem_ext out);
void softmax_tree(cl_mem_ext input, int spatial, int batch, int stride, float temp, cl_mem_ext output, tree hier);
void softmax_offset_tree(cl_mem_ext input, int offset, int spatial, int batch, int stride, float temp, cl_mem_ext output, tree hier);
void upsample_gpu(cl_mem_ext in, int w, int h, int c, int batch, int stride, int forward, float scale, cl_mem_ext out);

void softmax_offset_gpu(cl_mem_ext input, int offset, int n, int batch, int batch_offset, int groups, int group_offset, int stride, float temp, cl_mem_ext output);
void softmax_gpu(cl_mem_ext input, int n, int batch, int batch_offset, int groups, int group_offset, int stride, float temp, cl_mem_ext output);
void softmax_tree_gpu(cl_mem_ext input, int spatial, int batch, int stride, float temp, cl_mem_ext output, tree hier);

void upsample_gpu(cl_mem_ext in, int w, int h, int c, int batch, int stride, int forward, float scale, cl_mem_ext out);

void mean_array_gpu(cl_mem_ext src, int size, float alpha, cl_mem_ext avg);

int check_sim(size_t i, size_t j, contrastive_params *contrast_p, int contrast_p_size);
float find_sim(size_t i, size_t j, contrastive_params *contrast_p, int contrast_p_size);
float find_P_constrastive(size_t i, size_t j, contrastive_params *contrast_p, int contrast_p_size);
float P_constrastive_f_det(size_t il, int *labels, float **z, unsigned int feature_size, float temperature, contrastive_params *contrast_p, int contrast_p_size);
float P_constrastive_f(size_t i, size_t l, int *labels, float **z, unsigned int feature_size, float temperature, contrastive_params *contrast_p, int contrast_p_size);
void grad_contrastive_loss_positive_f(size_t i, int *class_ids, int *labels, size_t num_of_samples, float **z, unsigned int feature_size, float temperature, float *delta, int wh, contrastive_params *contrast_p, int contrast_p_size);
void grad_contrastive_loss_negative_f(size_t i, int *class_ids, int *labels, size_t num_of_samples, float **z, unsigned int feature_size, float temperature, float *delta, int wh, contrastive_params *contrast_p, int contrast_p_size, int neg_max);

float math_vector_length(float *A, unsigned int feature_size);
float cosine_similarity(float *A, float *B, unsigned int feature_size);
float P_constrastive(size_t i, size_t l, int *labels, size_t num_of_samples, float **z, unsigned int feature_size, float temperature, float *cos_sim, float *exp_cos_sim);
void grad_contrastive_loss_positive(size_t i, int *labels, size_t num_of_samples, float **z, unsigned int feature_size, float temperature, float *cos_sim, float *p_constrastive, float *delta, int wh);
void grad_contrastive_loss_negative(size_t i, int *labels, size_t num_of_samples, float **z, unsigned int feature_size, float temperature, float *cos_sim, float *p_constrastive, float *delta, int wh);

#endif
#endif
