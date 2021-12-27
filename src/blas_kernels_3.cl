#ifndef __BLAS_KERNELS_3_CL__
#define __BLAS_KERNELS_3_CL__

static const char* const blas_kernel_source_3 = CONVERT_KERNEL_TO_STRING(

__kernel void weighted_delta_kernel(int n, __global float *a, __global float *b, __global float *s, __global float *da, __global float *db, __global float *ds, __global float *dc)
{
    int i = (get_group_id(0) + get_group_id(1)*get_num_groups(0)) * get_local_size(0) + get_local_id(0);
    if(i < n){
        if(da) da[i] += dc[i] * s[i];
        db[i] += dc[i] * (1-s[i]);
        ds[i] += dc[i] * a[i] + dc[i] * -b[i];
    }
}


__kernel void mult_add_into_kernel(int n, __global float *a, __global float *b, __global float *c)
{
    int i = (get_group_id(0) + get_group_id(1)*get_num_groups(0)) * get_local_size(0) + get_local_id(0);
    if(i < n){
        c[i] += a[i]*b[i];
    }
}


__kernel void deinter_kernel(int NX, __global float *X, int NY, __global float *Y, int B, __global float *OUT)
{
    int i = (get_group_id(0) + get_group_id(1)*get_num_groups(0)) * get_local_size(0) + get_local_id(0);
    if(i < (NX+NY)*B){
        int b = i / (NX+NY);
        int j = i % (NX+NY);
        if (j < NX){
            if(X) X[b*NX + j] += OUT[i];
        } else {
            if(Y) Y[b*NY + j - NX] += OUT[i];
        }
    }
}


__kernel void inter_kernel(int NX, __global float *X, int NY, __global float *Y, int B, __global float *OUT)
{
    int i = (get_group_id(0) + get_group_id(1)*get_num_groups(0)) * get_local_size(0) + get_local_id(0);
    if(i < (NX+NY)*B){
        int b = i / (NX+NY);
        int j = i % (NX+NY);
        if (j < NX){
            OUT[i] = X[b*NX + j];
        } else {
            OUT[i] = Y[b*NY + j - NX];
        }
    }
}


__kernel void softmax_device(__global float *input, int n, float temp, int stride, __global float *output)
{
    int i;
    float sum = 0;
    float largest = -FLT_MAX;
    for(i = 0; i < n; ++i){
        int val = input[i*stride];
        largest = (val>largest) ? val : largest;
    }
    for(i = 0; i < n; ++i){
        float e = exp(input[i*stride]/temp - largest/temp);
        sum += e;
        output[i*stride] = e;
    }
    for(i = 0; i < n; ++i){
        output[i*stride] /= sum;
    }
}


__kernel void softmax_kernel(__global float *input, int offset, int n, int batch, int batch_offset, int groups, int group_offset, int stride, float temp, __global float *output)
{
    int id = (get_group_id(0) + get_group_id(1)*get_num_groups(0)) * get_local_size(0) + get_local_id(0);
    if (id >= batch*groups) return;
    int b = id / groups;
    int g = id % groups;
    softmax_device(input + b*batch_offset + g*group_offset + offset, n, temp, stride, output + b*batch_offset + g*group_offset + offset);
}


__kernel void softmax_tree_kernel(__global float *input, int offset, int index, int spatial, int batch, int stride, float temp, __global float *output, int groups, __global float *group_size, __global float *group_offset)
{
    int id = (get_group_id(0) + get_group_id(1)*get_num_groups(0)) * get_local_size(0) + get_local_id(0);
    if (id >= spatial*batch*groups) return;
    int s = id % spatial;
    id = id / spatial;
    int g = id % groups;
    int b = id / groups;
    int goff = group_offset[g]*spatial;
    int boff = b*stride;
    softmax_device(input + offset + goff + boff + s, group_size[g], temp, spatial, output + offset + goff + boff + s);
}


__kernel void scale_mask_kernel(int n, __global float *x, float mask_num, __global float *mask, float scale)
{
    int i = (get_group_id(0) + get_group_id(1)*get_num_groups(0)) * get_local_size(0) + get_local_id(0);
    if(i < n && mask[i] == mask_num) x[i] *= scale;
}


__kernel void dot_kernel(__global float *output, float scale, int batch, int n, int size, __global float *delta)
{
    int index = (get_group_id(0) + get_group_id(1)*get_num_groups(0)) * get_local_size(0) + get_local_id(0);

    int f1 = index / n;
    int f2 = index % n;
    if (f2 <= f1) return;

    float sum = 0;
    float norm1 = 0;
    float norm2 = 0;
    int b, i;
    for(b = 0; b <  batch; ++b){
        for(i = 0; i < size; ++i){
            int i1 = b * size * n + f1 * size + i;
            int i2 = b * size * n + f2 * size + i;
            sum += output[i1] * output[i2];
            norm1 += output[i1] * output[i1];
            norm2 += output[i2] * output[i2];
        }
    }
    norm1 = sqrt(fabs(norm1));
    norm2 = sqrt(fabs(norm2));
    float norm = norm1 * norm2;
    sum = sum / norm;
    for(b = 0; b <  batch; ++b){
        for(i = 0; i < size; ++i){
            int i1 = b * size * n + f1 * size + i;
            int i2 = b * size * n + f2 * size + i;
            delta[i1] += - scale * sum * output[i2] / norm;
            delta[i2] += - scale * sum * output[i1] / norm;
        }
    }
}


__kernel void upsample_kernel(int N, __global float *x, int w, int h, int c, int batch, int stride, int forward, float scale, __global float *out)
{
    int i = (get_group_id(0) + get_group_id(1)*get_num_groups(0)) * get_local_size(0) + get_local_id(0);
    if(i >= N) return;
    int out_index = i;
    int out_w = i%(w*stride);
    i = i/(w*stride);
    int out_h = i%(h*stride);
    i = i/(h*stride);
    int out_c = i%c;
    i = i/c;
    int b = i%batch;

    int in_w = out_w / stride;
    int in_h = out_h / stride;
    int in_c = out_c;

    int in_index = b*w*h*c + in_c*w*h + in_h*w + in_w;

    //if(forward) out[out_index] += scale * x[in_index];
    //else atomicAdd(&x[in_index], scale * out[out_index]);
    if(forward) out[out_index] += scale * x[in_index];
    else x[in_index] += scale * out[out_index];
}


__kernel void gemm_kernel(
        int tuning,
        __local float* sums,
        int TA, int TB,
        int M, int N, int K,
        __const float ALPHA,
        __global float *A, int offset_A, int lda,
        __global float *B, int offset_B, int ldb,
        __const float BETA,
        __global float *C, int offset_C, int ldc) {

    int td = get_global_id(0);
    int jN = get_global_id(1);
    int iM = get_global_id(2);

    __global float *a = &A[offset_A];
    __global float *b = &B[offset_B];
    __global float *c = &C[iM * ldc + jN + offset_C];

    c[0] *= BETA;

    barrier(CLK_LOCAL_MEM_FENCE);

    int kK = 0;
    float sum = 0;
    if (TA == 0 && TB == 0) {
        __global float *aK = &a[iM * lda + kK];
        __global float *bK = &b[jN + ldb * kK];
        for (int kK = td; kK < K; kK += tuning) {
            sum += ALPHA * aK[kK] * bK[ldb * kK];
        }
    } else if (TA == 1 && TB == 0) {
        __global float *aK = &a[iM + lda * kK];
        __global float *bK = &b[jN + ldb * kK];
        for (int kK = td; kK < K; kK += tuning) {
            sum += ALPHA * aK[lda * kK] * bK[ldb * kK];
        }
    } else if (TA == 0 && TB == 1) {
        __global float *aK = &a[iM * lda + kK];
        __global float *bK = &b[jN * ldb + kK];
        for (int kK = td; kK < K; kK += tuning) {
            sum += ALPHA * aK[kK] * bK[kK];
        }
    } else {
        __global float *aK = &a[iM + lda * kK];
        __global float *bK = &b[jN * ldb + kK];
        for (int kK = td; kK < K; kK += tuning) {
            sum += ALPHA * aK[lda * kK] * bK[kK];
        }
    }
    sums[td] = sum;

    barrier(CLK_LOCAL_MEM_FENCE);

    if (td == 0) {
        for (int ts = 0; ts < tuning; ++ts) {
            c[0] += sums[ts];
        }
    }
}


__kernel void scal_add_kernel(int N, float ALPHA, float BETA, __global float *X, int OFFX, int INCX)
{
    int i = (get_group_id(0) + get_group_id(1)*get_num_groups(0)) * get_local_size(0) + get_local_id(0);
    if (i < N) {
        X[i*INCX + OFFX] *= ALPHA;
        X[i*INCX + OFFX] += BETA;
    }
}

__kernel void mean_array_kernel(int N, float alpha, __global float *s, __global float *a)
{
    int i = (get_group_id(0) + get_group_id(1)*get_num_groups(0)) * get_local_size(0) + get_local_id(0);
    if (i >= N) return;
    a[i] *= (1 - alpha) + s[i];
    a[i] *= alpha;
    s[i] = a[i];
}

);
#endif