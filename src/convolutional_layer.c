#include "convolutional_layer.h"
#include "utils.h"
#include "batchnorm_layer.h"
#include "im2col.h"
#include "col2im.h"
#include "gemm.h"
#include <stdio.h>

#ifdef AI2
#include "xnor_layer.h"
#endif

void swap_binary(convolutional_layer *l)
{
	float *swap = l->weights;
	l->weights = l->binary_weights;
	l->binary_weights = swap;

#ifdef GPU
	if (gpu_index >= 0) {
		cl_mem_ext swap_gpu = l->weights_gpu;
		l->weights_gpu = l->binary_weights_gpu;
		l->binary_weights_gpu = swap_gpu;
	}
#endif
}

void binarize_weights(float *weights, int n, int size, float *binary)
{
	int i, f;
	for(f = 0; f < n; ++f){
		float mean = 0;
		for(i = 0; i < size; ++i){
			mean += fabs(weights[f*size + i]);
		}
		mean = mean / size;
		for(i = 0; i < size; ++i){
			binary[f*size + i] = (weights[f*size + i] > 0) ? mean : -mean;
		}
	}
}

void binarize_cpu(float *input, int n, float *binary)
{
	int i;
	for(i = 0; i < n; ++i){
		binary[i] = (input[i] > 0) ? 1 : -1;
	}
}

void binarize_input(float *input, int n, int size, float *binary)
{
	int i, s;
	for(s = 0; s < size; ++s){
		float mean = 0;
		for(i = 0; i < n; ++i){
			mean += fabs(input[i*size + s]);
		}
		mean = mean / n;
		for(i = 0; i < n; ++i){
			binary[i*size + s] = (input[i*size + s] > 0) ? mean : -mean;
		}
	}
}

int convolutional_out_height(convolutional_layer l)
{
	return (l.h + 2*l.pad - l.size) / l.stride + 1;
}

int convolutional_out_width(convolutional_layer l)
{
	return (l.w + 2*l.pad - l.size) / l.stride + 1;
}

image get_convolutional_image(convolutional_layer l)
{
	return float_to_image(l.out_w,l.out_h,l.out_c,l.output);
}

image get_convolutional_delta(convolutional_layer l)
{
	return float_to_image(l.out_w,l.out_h,l.out_c,l.delta);
}

static size_t get_workspace_size(layer l){
	return (size_t)l.out_h*l.out_w*l.size*l.size*l.c/l.groups;
}

convolutional_layer make_convolutional_layer(int batch, int h, int w, int c, int n, int groups, int size, int stride, int padding, ACTIVATION activation, int batch_normalize, int binary, int xnor, int adam)
{
	int i;
	convolutional_layer l = {0};
	l.type = CONVOLUTIONAL;

	l.groups = groups;
	l.h = h;
	l.w = w;
	l.c = c;
	l.n = n;
	l.binary = binary;
	l.xnor = xnor;
	l.batch = batch;
	l.stride = stride;
	l.size = size;
	l.pad = padding;
	l.batch_normalize = batch_normalize;

	l.weights = (float*)calloc(c/groups*n*size*size, sizeof(float));
	l.weight_updates = (float*)calloc(c/groups*n*size*size, sizeof(float));

	l.biases = (float*)calloc(n, sizeof(float));
	l.bias_updates = (float*)calloc(n, sizeof(float));

	l.nweights = c/groups*n*size*size;
	l.nbiases = n;

    //float scale = 1.727f/sqrtf(size*size*c);
	float scale = (float) sqrt(2. / (size * size * c / l.groups));
	//printf("convscale %f\n", scale);
	//float scale = .04f;
	for(i = 0; i < c*n*size*size; ++i) l.weights[i] = scale*rand_uniform(-1.f, 1.f);
	//for(i = 0; i < l.nweights; ++i) l.weights[i] = scale*rand_normal();
    //for(i = 0; i < l.nweights; ++i) l.weights[i] = scale*rand_uniform(-1.f, 1.f);
	int out_w = convolutional_out_width(l);
	int out_h = convolutional_out_height(l);
	l.out_h = out_h;
	l.out_w = out_w;
	l.out_c = n;
	l.outputs = l.out_h * l.out_w * l.out_c;
	l.inputs = l.w * l.h * l.c;

	l.output = (float*)calloc(l.batch*l.outputs, sizeof(float));
	l.delta  = (float*)calloc(l.batch*l.outputs, sizeof(float));

	l.forward = forward_convolutional_layer;
	l.backward = backward_convolutional_layer;
	l.update = update_convolutional_layer;
	if(binary){
		l.binary_weights = (float*)calloc(l.nweights, sizeof(float));
		l.cweights = (char*)calloc(l.nweights, sizeof(char));
		l.scales = (float*)calloc(n, sizeof(float));
	}
	if(xnor){
		l.binary_weights = (float*)calloc(l.nweights, sizeof(float));
		l.binary_input = (float*)calloc(l.inputs*l.batch, sizeof(float));
	}

	if(batch_normalize){
		l.scales = (float*)calloc(n, sizeof(float));
		l.scale_updates = (float*)calloc(n, sizeof(float));
		for(i = 0; i < n; ++i){
			l.scales[i] = 1;
		}

		l.mean = (float*)calloc(n, sizeof(float));
		l.variance = (float*)calloc(n, sizeof(float));

		l.mean_delta = (float*)calloc(n, sizeof(float));
		l.variance_delta = (float*)calloc(n, sizeof(float));

		l.rolling_mean = (float*)calloc(n, sizeof(float));
		l.rolling_variance = (float*)calloc(n, sizeof(float));

		l.x = (float*)calloc(l.batch*l.outputs, sizeof(float));
		l.x_norm = (float*)calloc(l.batch*l.outputs, sizeof(float));
	}
	if(adam){
		l.m = (float*)calloc(l.nweights, sizeof(float));
		l.v = (float*)calloc(l.nweights, sizeof(float));
		l.bias_m = (float*)calloc(n, sizeof(float));
		l.scale_m = (float*)calloc(n, sizeof(float));
		l.bias_v = (float*)calloc(n, sizeof(float));
		l.scale_v = (float*)calloc(n, sizeof(float));
	}

#ifdef GPU
	if (gpu_index >= 0) {
		l.forward_gpu = forward_convolutional_layer_gpu;
		l.backward_gpu = backward_convolutional_layer_gpu;
		l.update_gpu = update_convolutional_layer_gpu;

		if (adam) {
			l.m_gpu = opencl_make_array(l.m, l.nweights);
			l.v_gpu = opencl_make_array(l.v, l.nweights);
			l.bias_m_gpu = opencl_make_array(l.bias_m, n);
			l.bias_v_gpu = opencl_make_array(l.bias_v, n);
			l.scale_m_gpu = opencl_make_array(l.scale_m, n);
			l.scale_v_gpu = opencl_make_array(l.scale_v, n);
		}

		l.weights_gpu = opencl_make_array(l.weights, l.nweights);
		l.weight_updates_gpu = opencl_make_array(l.weight_updates, l.nweights);

		l.biases_gpu = opencl_make_array(l.biases, n);
		l.bias_updates_gpu = opencl_make_array(l.bias_updates, n);

		l.delta_gpu = opencl_make_array(l.delta, l.batch*out_h*out_w*n);
		l.output_gpu = opencl_make_array(l.output, l.batch*out_h*out_w*n);

		if(binary){
			l.binary_weights_gpu = opencl_make_array(l.weights, l.nweights);
		}
		if(xnor){
			l.binary_weights_gpu = opencl_make_array(l.weights, l.nweights);
			l.binary_input_gpu = opencl_make_array(l.binary_input, l.inputs*l.batch);
		}

		if(batch_normalize){
			l.mean_gpu = opencl_make_array(l.mean, n);
			l.variance_gpu = opencl_make_array(l.variance, n);

			l.rolling_mean_gpu = opencl_make_array(l.rolling_mean, n);
			l.rolling_variance_gpu = opencl_make_array(l.rolling_variance, n);

			l.mean_delta_gpu = opencl_make_array(l.mean_delta, n);
			l.variance_delta_gpu = opencl_make_array(l.variance_delta, n);

			l.scales_gpu = opencl_make_array(l.scales, n);
			l.scale_updates_gpu = opencl_make_array(l.scale_updates, n);

			l.x_gpu = opencl_make_array(l.x, l.batch*l.outputs);
			l.x_norm_gpu = opencl_make_array(l.x_norm, l.batch*l.outputs);
		}
	}
#endif
	l.workspace_size = get_workspace_size(l);
	l.activation = activation;

	fprintf(stderr, "conv  %5d %2d x%2d /%2d  %4d x%4d x%4d   ->  %4d x%4d x%4d  %5.3f BFLOPs\n", n, size, size, stride, w, h, c, l.out_w, l.out_h, l.out_c, (2.0 * l.n * l.size*l.size*l.c/l.groups * l.out_h*l.out_w)/1000000000.);

	return l;
}

void denormalize_convolutional_layer(convolutional_layer l)
{
    int i, j;
    for(i = 0; i < l.n; ++i){
        float scale = l.scales[i]/sqrt(l.rolling_variance[i] + .00001);
        for(j = 0; j < l.c/l.groups*l.size*l.size; ++j){
            l.weights[i*l.c/l.groups*l.size*l.size + j] *= scale;
        }
        l.biases[i] -= l.rolling_mean[i] * scale;
        l.scales[i] = 1;
        l.rolling_mean[i] = 0;
        l.rolling_variance[i] = 1;
    }
}

/*
void test_convolutional_layer()
{
	convolutional_layer l = make_convolutional_layer(1, 5, 5, 3, 2, 5, 2, 1, LEAKY, 1, 0, 0, 0);
	l.batch_normalize = 1;
	float data[] = {1,1,1,1,1,
		1,1,1,1,1,
		1,1,1,1,1,
		1,1,1,1,1,
		1,1,1,1,1,
		2,2,2,2,2,
		2,2,2,2,2,
		2,2,2,2,2,
		2,2,2,2,2,
		2,2,2,2,2,
		3,3,3,3,3,
		3,3,3,3,3,
		3,3,3,3,3,
		3,3,3,3,3,
		3,3,3,3,3};
	//net.input = data;
	//forward_convolutional_layer(l);
}
*/

void resize_convolutional_layer(convolutional_layer *l, int w, int h)
{
#ifdef GPU
	if (gpu_index >= 0) {
		if (l->delta_gpu.ptr) opencl_free_gpu_only(l->delta_gpu);
		if (l->output_gpu.ptr) opencl_free_gpu_only(l->output_gpu);
		if (l->batch_normalize) {
			opencl_free_gpu_only(l->x_gpu);
			opencl_free_gpu_only(l->x_norm_gpu);
		}
	}
#endif
	l->w = w;
	l->h = h;
	int out_w = convolutional_out_width(*l);
	int out_h = convolutional_out_height(*l);
	l->out_w = out_w;
	l->out_h = out_h;
	l->outputs = l->out_h * l->out_w * l->out_c;
	l->inputs = l->w * l->h * l->c;
	l->output = (float*)realloc(l->output, l->batch*l->outputs*sizeof(float));
	l->delta  = (float*)realloc(l->delta, l->batch*l->outputs*sizeof(float));
	if(l->batch_normalize){
		l->x = (float*)realloc(l->x, l->batch*l->outputs*sizeof(float));
		l->x_norm  = (float*)realloc(l->x_norm, l->batch*l->outputs*sizeof(float));
	}
#ifdef GPU
	if (gpu_index >= 0) {
		l->delta_gpu = opencl_make_array(l->delta, l->batch * l->outputs);
		l->output_gpu = opencl_make_array(l->output, l->batch * l->outputs);
		if (l->batch_normalize) {
			l->x_gpu = opencl_make_array(l->x, l->batch * l->outputs);
			l->x_norm_gpu = opencl_make_array(l->x_norm, l->batch * l->outputs);
		}
	}
#endif
	l->workspace_size = get_workspace_size(*l);
}

void add_bias(float *output, float *biases, int batch, int n, int size)
{
	int i,j,b;
	for(b = 0; b < batch; ++b){
		for(i = 0; i < n; ++i){
			for(j = 0; j < size; ++j){
				output[(b*n + i)*size + j] += biases[i];
			}
		}
	}
}

void scale_bias(float *output, float *scales, int batch, int n, int size)
{
    int i,j,b;
    for(b = 0; b < batch; ++b){
        for(i = 0; i < n; ++i){
            for(j = 0; j < size; ++j){
                output[(b*n + i)*size + j] *= scales[i];
            }
        }
    }
}

void backward_bias(float *bias_updates, float *delta, int batch, int n, int size)
{
	int i,b;
	for(b = 0; b < batch; ++b){
		for(i = 0; i < n; ++i){
			bias_updates[i] += sum_array(delta+size*(i+b*n), size);
		}
	}
}

void forward_convolutional_layer(convolutional_layer l, network net)
{
	int i, j;

	fill_cpu(l.outputs*l.batch, 0, l.output, 1);

	if(l.xnor){
		binarize_weights(l.weights, l.n, l.c/l.groups*l.size*l.size, l.binary_weights);
		swap_binary(&l);
		binarize_cpu(net.input, l.c*l.h*l.w*l.batch, l.binary_input);
		net.input = l.binary_input;
	}

	int m = l.n/l.groups;
	int k = l.size*l.size*l.c/l.groups;
	int n = l.out_w*l.out_h;
	for(i = 0; i < l.batch; ++i){
		for(j = 0; j < l.groups; ++j){
			float *a = l.weights + j*l.nweights/l.groups;
			float *b = net.workspace;
			float *c = l.output + (i*l.groups + j)*n*m;
			float *im =  net.input + (i*l.groups + j)*l.c/l.groups*l.h*l.w;

			if (l.size == 1) {
				b = im;
			} else {
				im2col_cpu(im, l.c/l.groups, l.h, l.w, l.size, l.stride, l.pad, b);
			}
			gemm(0,0,m,n,k,1,a,k,b,n,1,c,n);
		}
	}

	if(l.batch_normalize){
		forward_batchnorm_layer(l, net);
	} else {
		add_bias(l.output, l.biases, l.batch, l.n, l.out_h*l.out_w);
	}

	activate_array(l.output, l.outputs*l.batch, l.activation);
	if(l.binary || l.xnor) swap_binary(&l);
}

void backward_convolutional_layer(convolutional_layer l, network net)
{
	int i, j;
	int m = l.n/l.groups;
	int n = l.size*l.size*l.c/l.groups;
	int k = l.out_w*l.out_h;

	gradient_array(l.output, l.outputs*l.batch, l.activation, l.delta);

	if(l.batch_normalize){
		backward_batchnorm_layer(l, net);
	} else {
		backward_bias(l.bias_updates, l.delta, l.batch, l.n, k);
	}

	for(i = 0; i < l.batch; ++i){
		for(j = 0; j < l.groups; ++j){
			float *a = l.delta + (i*l.groups + j)*m*k;
			float *b = net.workspace;
			float *c = l.weight_updates + j*l.nweights/l.groups;

			float *im  = net.input + (i*l.groups + j)*l.c/l.groups*l.h*l.w;
			float *imd = net.delta + (i*l.groups + j)*l.c/l.groups*l.h*l.w;

			if(l.size == 1){
				b = im;
			} else {
				im2col_cpu(im, l.c/l.groups, l.h, l.w,
						l.size, l.stride, l.pad, b);
			}

			gemm(0,1,m,n,k,1,a,k,b,k,1,c,n);

			if (net.delta) {
				a = l.weights + j*l.nweights/l.groups;
				b = l.delta + (i*l.groups + j)*m*k;
				c = net.workspace;
				if (l.size == 1) {
					c = imd;
				}

				gemm(1,0,n,k,m,1,a,n,b,k,0,c,k);

				if (l.size != 1) {
					col2im_cpu(net.workspace, l.c/l.groups, l.h, l.w, l.size, l.stride, l.pad, imd);
				}
			}
		}
	}
}

void update_convolutional_layer(convolutional_layer l, update_args a)
{
	float learning_rate = a.learning_rate*l.learning_rate_scale;
	float momentum = a.momentum;
	float decay = a.decay;
	int batch = a.batch;

	axpy_cpu(l.n, learning_rate/batch, l.bias_updates, 1, l.biases, 1);
	scal_cpu(l.n, momentum, l.bias_updates, 1);

	if(l.scales){
		axpy_cpu(l.n, learning_rate/batch, l.scale_updates, 1, l.scales, 1);
		scal_cpu(l.n, momentum, l.scale_updates, 1);
	}

	axpy_cpu(l.nweights, -decay*batch, l.weights, 1, l.weight_updates, 1);
	axpy_cpu(l.nweights, learning_rate/batch, l.weight_updates, 1, l.weights, 1);
	scal_cpu(l.nweights, momentum, l.weight_updates, 1);
}


image get_convolutional_weight(convolutional_layer l, int i)
{
	int h = l.size;
	int w = l.size;
	int c = l.c/l.groups;
	return float_to_image(w,h,c,l.weights+i*h*w*c);
}

void rgbgr_weights(convolutional_layer l)
{
	int i;
	for(i = 0; i < l.n; ++i){
		image im = get_convolutional_weight(l, i);
		if (im.c == 3) {
			rgbgr_image(im);
		}
	}
}

void rescale_weights(convolutional_layer l, float scale, float trans)
{
	int i;
	for(i = 0; i < l.n; ++i){
		image im = get_convolutional_weight(l, i);
		if (im.c == 3) {
			scale_image(im, scale);
			float sum = sum_array(im.data, im.w*im.h*im.c);
			l.biases[i] += sum*trans;
		}
	}
}

image *get_weights(convolutional_layer l)
{
	image *weights = (image*)calloc(l.n, sizeof(image));
	int i;
	for(i = 0; i < l.n; ++i){
		weights[i] = copy_image(get_convolutional_weight(l, i));
		normalize_image(weights[i]);
		/*
		   char buff[256];
		   sprintf(buff, "filter%d", i);
		   save_image(weights[i], buff);
		 */
	}
	//error("hey");
	return weights;
}

image *visualize_convolutional_layer(convolutional_layer l, char *window, image *prev_weights)
{
	image *single_weights = get_weights(l);
	show_images(single_weights, l.n, window);

	image delta = get_convolutional_image(l);
	image dc = collapse_image_layers(delta, 1);
	char buff[256];
	sprintf(buff, "%s: Output", window);
	//show_image(dc, buff);
	//save_image(dc, buff);
	free_image(dc);
	return single_weights;
}

