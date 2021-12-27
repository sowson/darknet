#include <stdio.h>
#include <time.h>
#include <assert.h>
#include "darknet.h"
#include "network.h"
#include "data.h"
#include "opencl.h"
#include "crop_layer.h"
#include "convolutional_layer.h"
#include "region_layer.h"
#include "yolo_layer.h"
#include "yolo4_layer.h"
#include "normalization_layer.h"
#include "maxpool_layer.h"
#include "reorg_layer.h"
#include "avgpool_layer.h"
#include "cost_layer.h"
#include "route_layer.h"
#include "upsample_layer.h"
#include "shortcut_layer.h"

load_args get_base_args(network *net)
{
	load_args args = {0};
	args.w = net->w;
	args.h = net->h;
	args.size = net->w;

	args.min = net->min_crop;
	args.max = net->max_crop;
	args.angle = net->angle;
	args.aspect = net->aspect;
	args.exposure = net->exposure;
	args.center = net->center;
	args.saturation = net->saturation;
	args.hue = net->hue;
	return args;
}

network *load_network(char *cfg, char *weights, int clear)
{
	network *net = parse_network_cfg(cfg);
	if(weights && weights[0] != 0){
		load_weights(net, weights);
	}
	if(clear) (*net->seen) = 0;
	return net;
}

int64_t get_current_iteration(network net)
{
    return *net.cur_iteration;
}

size_t get_current_batch(network *net)
{
	size_t batch_num = (*net->seen)/(net->batch*net->subdivisions);
	return batch_num;
}

void reset_network_state(network *net, int b)
{
	int i;
	for (i = 0; i < net->n; ++i) {
#ifdef GPU
		if (gpu_index >= 0) {
            opencl_set_device(net->gpu_index);
			layer l = net->layers[i];
			if (l.state_gpu.ptr) {
				fill_offset_gpu(l.outputs, 0, l.state_gpu, l.outputs * b, 1);
			}
			if (l.h_gpu.ptr) {
				fill_offset_gpu(l.outputs, 0, l.h_gpu, l.outputs * b, 1);
			}
		}
#endif
	}
}

void reset_rnn(network *net)
{
	reset_network_state(net, 0);
}

float get_current_rate(network *net)
{
    size_t batch_num = get_current_batch(net);
    int i;
    float rate;
    if (batch_num < net->burn_in) return net->learning_rate * pow((float)batch_num / net->burn_in, net->power);
    switch (net->policy) {
        case CONSTANT:
            return net->learning_rate;
        case STEP:
            return net->learning_rate * pow(net->scale, batch_num/net->step);
        case STEPS:
            rate = net->learning_rate;
            for(i = 0; i < net->num_steps; ++i){
                if(net->steps[i] > batch_num) return rate;
                rate *= net->scales[i];
            }
            return rate;
        case EXP:
            return net->learning_rate * pow(net->gamma, batch_num);
        case POLY:
            return net->learning_rate * pow(1 - (float)batch_num / net->max_batches, net->power);
        case RANDOM:
            return net->learning_rate * pow(rand_uniform(0,1), net->power);
        case SIG:
            return net->learning_rate * (1./(1.+exp(net->gamma*(batch_num - net->step))));
        case SGDR:
        {
            int last_iteration_start = 0;
            int cycle_size = net->batches_per_cycle;
            while ((last_iteration_start + cycle_size) < batch_num)
            {
                last_iteration_start += cycle_size;
                cycle_size *= net->batches_cycle_mult;
            }
            rate = net->learning_rate_min +
                   0.5*(net->learning_rate - net->learning_rate_min)
                   * (1. + cos((float)(batch_num - last_iteration_start)*3.14159265 / cycle_size));

            return rate;
        }
        default:
            fprintf(stderr, "Policy is weird!\n");
            return net->learning_rate;
    }
}

char *get_layer_string(LAYER_TYPE a)
{
	switch(a){
		case CONVOLUTIONAL:
			return "convolutional";
		case ACTIVE:
			return "activation";
		case LOCAL:
			return "local";
		case DECONVOLUTIONAL:
			return "deconvolutional";
		case CONNECTED:
			return "connected";
		case RNN:
			return "rnn";
		case GRU:
			return "gru";
		case LSTM:
			return "lstm";
		case CRNN:
			return "crnn";
		case MAXPOOL:
			return "maxpool";
		case REORG:
			return "reorg";
		case AVGPOOL:
			return "avgpool";
		case SOFTMAX:
			return "softmax";
		case DETECTION:
			return "detection";
		case REGION:
			return "region";
		case YOLO:
			return "yolo";
        case YOLO4:
            return "yolo4";
        case DROPOUT:
			return "dropout";
		case CROP:
			return "crop";
		case COST:
			return "cost";
		case ROUTE:
			return "route";
		case SHORTCUT:
			return "shortcut";
		case NORMALIZATION:
			return "normalization";
		case BATCHNORM:
			return "batchnorm";
		default:
			break;
	}
	return "none";
}

network *make_network(int n)
{
	network *net = (network*)calloc(1, sizeof(network));
	net->n = n;
	net->layers = (layer*)calloc(net->n, sizeof(layer));
	net->seen = (size_t*)calloc(1, sizeof(size_t));
	net->t    = (int*)calloc(1, sizeof(int));
	net->cost = (float*)calloc(1, sizeof(float));
    net->cur_iteration = (int*)calloc(1, sizeof(int));
    net->total_bbox = (int*)calloc(1, sizeof(int));
    net->rewritten_bbox = (int*)calloc(1, sizeof(int));
    *net->rewritten_bbox = *net->total_bbox = 0;
#ifdef GPU
	if (gpu_index >= 0) {
        net->delta_gpu.ptr = NULL;
        net->output_gpu.ptr = NULL;
        net->truth_gpu.ptr = NULL;
        net->input_gpu.ptr = NULL;
        net->workspace_gpu.ptr = NULL;

        int l;
        for(l = 0; l < n; ++l) {
            net->layers[l].indexes_gpu.ptr = NULL;
            net->layers[l].z_gpu.ptr = NULL;
            net->layers[l].r_gpu.ptr = NULL;
            net->layers[l].h_gpu.ptr = NULL;
            net->layers[l].temp_gpu.ptr = NULL;
            net->layers[l].temp2_gpu.ptr = NULL;
            net->layers[l].temp3_gpu.ptr = NULL;
            net->layers[l].dh_gpu.ptr = NULL;
            net->layers[l].hh_gpu.ptr = NULL;
            net->layers[l].prev_cell_gpu.ptr = NULL;
            net->layers[l].cell_gpu.ptr = NULL;
            net->layers[l].f_gpu.ptr = NULL;
            net->layers[l].i_gpu.ptr = NULL;
            net->layers[l].g_gpu.ptr = NULL;
            net->layers[l].o_gpu.ptr = NULL;
            net->layers[l].c_gpu.ptr = NULL;
            net->layers[l].dc_gpu.ptr = NULL;
            net->layers[l].m_gpu.ptr = NULL;
            net->layers[l].v_gpu.ptr = NULL;
            net->layers[l].bias_m_gpu.ptr = NULL;
            net->layers[l].scale_m_gpu.ptr = NULL;
            net->layers[l].bias_v_gpu.ptr = NULL;
            net->layers[l].scale_v_gpu.ptr = NULL;
            net->layers[l].combine_gpu.ptr = NULL;
            net->layers[l].combine_delta_gpu.ptr = NULL;
            net->layers[l].prev_state_gpu.ptr = NULL;
            net->layers[l].forgot_state_gpu.ptr = NULL;
            net->layers[l].forgot_delta_gpu.ptr = NULL;
            net->layers[l].state_gpu.ptr = NULL;
            net->layers[l].state_delta_gpu.ptr = NULL;
            net->layers[l].concat_gpu.ptr = NULL;
            net->layers[l].concat_delta_gpu.ptr = NULL;
            net->layers[l].binary_input_gpu.ptr = NULL;
            net->layers[l].binary_weights_gpu.ptr = NULL;
            net->layers[l].mean_gpu.ptr = NULL;
            net->layers[l].variance_gpu.ptr = NULL;
            net->layers[l].rolling_mean_gpu.ptr = NULL;
            net->layers[l].rolling_variance_gpu.ptr = NULL;
            net->layers[l].variance_delta_gpu.ptr = NULL;
            net->layers[l].mean_delta_gpu.ptr = NULL;
            net->layers[l].x_gpu.ptr = NULL;
            net->layers[l].x_norm_gpu.ptr = NULL;
            net->layers[l].weights_gpu.ptr = NULL;
            net->layers[l].weight_updates_gpu.ptr = NULL;
            net->layers[l].weight_change_gpu.ptr = NULL;
            net->layers[l].biases_gpu.ptr = NULL;
            net->layers[l].bias_updates_gpu.ptr = NULL;
            net->layers[l].bias_change_gpu.ptr = NULL;
            net->layers[l].scales_gpu.ptr = NULL;
            net->layers[l].scale_updates_gpu.ptr = NULL;
            net->layers[l].scale_change_gpu.ptr = NULL;
            net->layers[l].output_gpu.ptr = NULL;
            net->layers[l].loss_gpu.ptr = NULL;
            net->layers[l].delta_gpu.ptr = NULL;
            net->layers[l].rand_gpu.ptr = NULL;
            net->layers[l].squared_gpu.ptr = NULL;
            net->layers[l].norms_gpu.ptr = NULL;
        }
    }
#endif
	return net;
}


void forward_network(network *netp)
{
#ifdef GPU
    if(netp->gpu_index >= 0){
        forward_network_gpu(netp);   
        return;
    }
#endif
    network net = *netp;
    int i;
    for(i = 0; i < net.n; ++i){
        net.index = i;
        layer l = net.layers[i];
        if(l.delta){
            fill_cpu(l.outputs * l.batch, 0, l.delta, 1);
        }
#ifdef BENCHMARK
        clock_t t;
		t = clock();
#endif
        l.forward(l, net);
#ifdef BENCHMARK
        t = clock() - t;
		double time_taken = ((double)t);
		const char* layerName[] = { "CONVOLUTIONAL","DECONVOLUTIONAL","CONNECTED", "MAXPOOL", "SOFTMAX", "DETECTION", "DROPOUT", "CROP", "ROUTE", "COST", "NORMALIZATION", "AVGPOOL", "LOCAL", "SHORTCUT", "ACTIVE", "RNN", "GRU", "LSTM", "CRNN", "BATCHNORM", "NETWORK", "XNOR", "REGION", "YOLO", "YOLO4", "ISEG", "REORG", "UPSAMPLE", "LOGXENT", "L2NORM", "BLANK"};
		printf("FW %s\t%d\n", layerName[(int)l.type], (int)time_taken);
#endif
        net.input = l.output;
        if(l.truth) {
            net.truth = l.output;
        }
    }
    calc_network_cost(netp);
}

void update_network(network *netp)
{
#ifdef GPU
    if(netp->gpu_index >= 0){
        update_network_gpu(netp);   
        return;
    }
#endif
	network net = *netp;
	int i;
	update_args a = {0};
	a.batch = net.batch*net.subdivisions;
	a.learning_rate = get_current_rate(netp);
	a.momentum = net.momentum;
	a.decay = net.decay;
	a.adam = net.adam;
	a.B1 = net.B1;
	a.B2 = net.B2;
	a.eps = net.eps;
	++*net.t;
	a.t = *net.t;

	for(i = 0; i < net.n; ++i){
		layer l = net.layers[i];
		if(l.update){
#ifdef BENCHMARK
		clock_t t;
		t = clock();
#endif
            l.update(l, a);
#ifdef BENCHMARK
        t = clock() - t;
		double time_taken = ((double)t);
		const char* layerName[] = { "CONVOLUTIONAL","DECONVOLUTIONAL","CONNECTED", "MAXPOOL", "SOFTMAX", "DETECTION", "DROPOUT", "CROP", "ROUTE", "COST", "NORMALIZATION", "AVGPOOL", "LOCAL", "SHORTCUT", "ACTIVE", "RNN", "GRU", "LSTM", "CRNN", "BATCHNORM", "NETWORK", "XNOR", "REGION", "YOLO", "YOLO4", "ISEG", "REORG", "UPSAMPLE", "LOGXENT", "L2NORM", "BLANK"};
		printf("UP %s\t%d\n", layerName[(int)l.type], (int)time_taken);
#endif
        }
	}
}

void calc_network_cost(network *netp)
{
	network net = *netp;
	int i;
	float sum = 0;
	int count = 0;
	for(i = 0; i < net.n; ++i){
		if(net.layers[i].cost){
			sum += net.layers[i].cost[0];
			++count;
		}
	}
	*net.cost = sum/count;
}

int get_predicted_class_network(network *net)
{
	return max_index(net->output, net->outputs);
}

void backward_network(network *netp)
{
#ifdef GPU
    if(netp->gpu_index >= 0){
        backward_network_gpu(netp);   
        return;
    }
#endif
	network net = *netp;
	int i;
	network orig = net;
	for(i = net.n-1; i >= 0; --i){
		layer l = net.layers[i];
		if(l.stopbackward) break;
		if(i == 0){
			net = orig;
		}else{
			layer prev = net.layers[i-1];
			net.input = prev.output;
			net.delta = prev.delta;
		}
		net.index = i;
#ifdef BENCHMARK
        clock_t t;
		t = clock();
#endif
        l.backward(l, net);
#ifdef BENCHMARK
        t = clock() - t;
		double time_taken = ((double)t);
		const char* layerName[] = { "CONVOLUTIONAL","DECONVOLUTIONAL","CONNECTED", "MAXPOOL", "SOFTMAX", "DETECTION", "DROPOUT", "CROP", "ROUTE", "COST", "NORMALIZATION", "AVGPOOL", "LOCAL", "SHORTCUT", "ACTIVE", "RNN", "GRU", "LSTM", "CRNN", "BATCHNORM", "NETWORK", "XNOR", "REGION", "YOLO", "YOLO4", "ISEG", "REORG", "UPSAMPLE", "LOGXENT", "L2NORM", "BLANK"};
		printf("BW %s\t%d\n", layerName[(int)l.type], (int)time_taken);
#endif
	}
}

float train_network_datum(network *net)
{
	*net->seen += net->batch;
	net->train = 1;
	forward_network(net);
	backward_network(net);
	float error = *net->cost;
	if(((*net->seen)/net->batch)%net->subdivisions == 0) update_network(net);
	return error;
}

float train_network_datum_cgan(network *net, data d, data o, int i)
{
    int batch = net->batch;
    int n = d.X.rows / batch;

    assert(batch == 1);

    float error = 0;

    *net->seen += net->batch;
    net->train = 1;

    data p;

    if (*net->seen % 2 == 0) {
        p = copy_data(d);
        get_next_batch(d, batch, i*batch, net->input, net->truth);
    }
    else {
        p = copy_data(o);
        get_next_batch(o, batch, i*batch, net->input, net->truth);
    }
    forward_network(net);

    layer lout = get_network_output_layer_cgan(net);

    assert(net->inputs == net->outputs);
    get_next_batch(p, batch, i*batch, net->output, 0);
    opencl_push_array(net->output_gpu, net->output, lout.outputs*lout.batch);

    backward_network(net);

    error += *net->cost;
    if((*net->seen) % 32 == 0) update_network(net);

    free_data(p);

    return error;
}

float train_network_sgd(network *net, data d, int n)
{
	int batch = net->batch;

	int i;
	float sum = 0;
	for(i = 0; i < n; ++i){
		get_random_batch(d, batch, net->input, net->truth);
		float err = train_network_datum(net);
		sum += err;
	}
    return (float)sum/(n*batch);
}

float train_network(network *net, data d)
{
	assert(d.X.rows % net->batch == 0);
	int batch = net->batch;
	int n = d.X.rows / batch;

	int i;
	float sum = 0;
	for(i = 0; i < n; ++i){
#if GPU
	    if (gpu_index >= 0) {
            opencl_set_device(net->gpu_index);
        }
#endif
		get_next_batch(d, batch, i*batch, net->input, net->truth);
		float err = train_network_datum(net);
		sum += err;
	}
    return (float)sum/(n*batch);
}

float train_network_cgan(network *net, data d, data o)
{
    assert(d.X.rows % net->batch == 0);
    assert(o.X.rows % net->batch == 0);
    int batch = net->batch;
    int n = d.X.rows / batch;

    int i;
    float sum = 0;
    for(i = 0; i < n; ++i){
        float err = 0;
        err += train_network_datum_cgan(net, d, o, i);
        sum += err / 2.0f;
    }
    return sum / (n * batch);
}

void set_temp_network(network *net, float t)
{
	int i;
	for(i = 0; i < net->n; ++i){
		net->layers[i].temperature = t;
	}
}


void set_batch_network(network *net, int b)
{
	net->batch = b;
	int i;
	for(i = 0; i < net->n; ++i){
		net->layers[i].batch = b;
	}
}

int resize_network(network *net, int w, int h)
{
	int i;
	//if(w == net->w && h == net->h) return 0;
	net->w = w;
	net->h = h;
	int inputs = 0;
	size_t workspace_size = 0;
	//fprintf(stderr, "Resizing to %d x %d...\n", w, h);
	//fflush(stderr);
	for (i = 0; i < net->n; ++i){
		layer l = net->layers[i];
		if(l.type == CONVOLUTIONAL ||
		   l.type == DECONVOLUTIONAL){
			resize_convolutional_layer(&l, w, h);
		}else if(l.type == CROP){
			resize_crop_layer(&l, w, h);
		}else if(l.type == MAXPOOL){
			resize_maxpool_layer(&l, w, h);
		}else if(l.type == REGION){
			resize_region_layer(&l, w, h);
		}else if(l.type == YOLO){
			resize_yolo_layer(&l, w, h);
        }else if(l.type == YOLO4){
			resize_yolo4_layer(&l, w, h);
		}else if(l.type == ROUTE){
			resize_route_layer(&l, net);
		}else if(l.type == SHORTCUT){
			resize_shortcut_layer(&l, w, h);
		}else if(l.type == UPSAMPLE){
			resize_upsample_layer(&l, w, h);
		}else if(l.type == REORG){
			resize_reorg_layer(&l, w, h);
		}else if(l.type == AVGPOOL){
			resize_avgpool_layer(&l, w, h);
		}else if(l.type == NORMALIZATION){
			resize_normalization_layer(&l, w, h);
		}else if(l.type == COST){
			resize_cost_layer(&l, inputs);
		}else{
			error("Cannot resize this type of layer");
		}
		if(l.workspace_size > workspace_size) workspace_size = l.workspace_size;
		if(l.workspace_size > 2000000000) assert(0);
		inputs = l.outputs;
		net->layers[i] = l;
		w = l.out_w;
		h = l.out_h;
		if(l.type == AVGPOOL) break;
	}
	layer out = get_network_output_layer(net);
	net->inputs = net->layers[0].inputs;
	net->outputs = out.outputs;
	net->truths = out.outputs;
	if(net->layers[net->n-1].truths) net->truths = net->layers[net->n-1].truths;
    net->output = out.output;
    free(net->input);
    free(net->truth);
    net->input = (float*)calloc(net->inputs*net->batch, sizeof(float));
    net->truth = (float*)calloc(net->truths*net->batch, sizeof(float));
#ifdef GPU
	if (gpu_index >= 0) {
		net->output_gpu = out.output_gpu;
		opencl_free_gpu_only(net->input_gpu);
		opencl_free_gpu_only(net->truth_gpu);
		net->input_gpu = opencl_make_array(net->input, net->inputs * net->batch);
		net->truth_gpu = opencl_make_array(net->truth, net->truths * net->batch);
		//TODO: CHECK! (2)
        //opencl_free_gpu_only(net->delta_gpu);
        //net->delta_gpu = opencl_make_array(net->delta, net->outputs * net->batch);
		opencl_free(net->workspace_gpu);
		if(workspace_size){
		    net->workspace = (float*)calloc(workspace_size, sizeof(float));
		    net->workspace_gpu = opencl_make_array(net->workspace, workspace_size);
		}
	}
	else {
		free(net->workspace);
		net->workspace = (float*)calloc(workspace_size, sizeof(float));
	}
#else
	free(net->workspace);
	net->workspace = calloc(workspace_size, sizeof(float));
#endif
	//fprintf(stderr, " Done!\n");
	return 0;
}

layer get_network_detection_layer(network *net)
{
	int i;
	for(i = 0; i < net->n; ++i){
		if(net->layers[i].type == DETECTION){
			return net->layers[i];
		}
	}
	fprintf(stderr, "Detection layer not found!!\n");
	layer l = {0};
	return l;
}

int get_network_output_size(network net)
{
    int i;
    for(i = net.n-1; i > 0; --i) if(net.layers[i].type != COST) break;
    return net.layers[i].outputs;
}

int get_network_input_size(network net)
{
    return net.layers[0].inputs;
}

image get_network_image_layer(network *net, int i)
{
	layer l = net->layers[i];
#ifdef GPU
	if (gpu_index >= 0) {
		//opencl_pull_array(l.output_gpu, l.output, l.outputs);
	}
#endif
	if (l.out_w && l.out_h && l.out_c){
		return float_to_image(l.out_w, l.out_h, l.out_c, l.output);
	}
	image def = {0};
	return def;
}

image get_network_image(network *net)
{
	int i;
	for(i = net->n-1; i >= 0; --i){
		image m = get_network_image_layer(net, i);
		if(m.h != 0) return m;
	}
	image def = {0};
	return def;
}

void visualize_network(network *net)
{
	image *prev = 0;
	int i;
	char buff[256];
	for(i = 0; i < net->n; ++i){
		sprintf(buff, "Layer %d", i);
		layer l = net->layers[i];
		if(l.type == CONVOLUTIONAL){
			prev = visualize_convolutional_layer(l, buff, prev);
		}
	}
}

void top_predictions(network *net, int k, int *index)
{
	top_k(net->output, net->outputs, k, index);
}


float *network_predict(network *net, float *input)
{
	network orig = *net;
	//net->input = input;
    memcpy(net->input, input, net->inputs * net->batch * sizeof(float));
	net->truth = 0;
	net->train = 0;
	net->delta = 0;
	forward_network(net);
	float *out = net->output;
	*net = orig;
	return out;
}

#ifdef GPU
float *get_network_output_layer_gpu(network net, int i)
{
    layer l = net.layers[i];
    if(l.type != REGION && l.type != YOLO && l.type != YOLO4) opencl_pull_array(l.output_gpu, l.output, l.outputs*l.batch);
    return l.output;
}

float *get_network_output_gpu(network net)
{
    int i;
    for(i = net.n-1; i > 0; --i) if(net.layers[i].type != COST) break;
    return get_network_output_layer_gpu(net, i);
}
#endif

float *get_network_output_y4(network net)
{
#ifdef GPU
    if (gpu_index >= 0) return get_network_output_gpu(net);
#endif
    int i;
    for(i = net.n-1; i > 0; --i) if(net.layers[i].type != COST) break;
    return net.layers[i].output;
}

float *network_predict_y4(network *net, float *input)
{
    net->index = 0;
    //net->input = input;
    memcpy(net->input, input, net->inputs * net->batch * sizeof(float));
    net->truth = 0;
    net->train = 0;
    net->delta = 0;
    forward_network(net);
    float *out = get_network_output_y4(*net);
    return out;
}

int num_detections(network *net, float thresh)
{
	int i;
	int s = 0;
	for(i = 0; i < net->n; ++i){
		layer l = net->layers[i];
		if(l.type == YOLO){
			s += yolo_num_detections(l, thresh);
		}
		if(l.type == YOLO4){
			s += yolo4_num_detections(l, thresh);
		}
		if(l.type == DETECTION || l.type == REGION){
			s += l.w*l.h*l.n;
		}
	}
	return s;
}

detection *make_network_boxes(network *net, float thresh, int *num)
{
	layer l = net->layers[net->n - 1];
	int i;
	int nboxes = num_detections(net, thresh);
	if(num) *num = nboxes;
	detection *dets = (detection*)calloc(nboxes, sizeof(detection));
	for(i = 0; i < nboxes; ++i){
		dets[i].prob = (float*)calloc(l.classes, sizeof(float));
		if(l.coords > 4){
			dets[i].mask = (float*)calloc(l.coords-4, sizeof(float));
		}
	}
	return dets;
}

void fill_network_boxes(network *net, int w, int h, float thresh, float hier, int *map, int relative, detection *dets)
{
	int j;
	for(j = 0; j < net->n; ++j){
		layer l = net->layers[j];
		if(l.type == YOLO){
			int count = get_yolo_detections(l, w, h, net->w, net->h, thresh, map, relative, dets);
			dets += count;
		}
		if(l.type == YOLO4){
			int count = get_yolo4_detections(l, w, h, net->w, net->h, thresh, map, relative, dets, 1);
			dets += count;
		}
		if(l.type == REGION){
			get_region_detections(l, w, h, net->w, net->h, thresh, map, hier, relative, dets);
			dets += l.w*l.h*l.n;
		}
		if(l.type == DETECTION){
			get_detection_detections(l, w, h, thresh, dets);
			dets += l.w*l.h*l.n;
		}
	}
}

int num_detections_y4(network *net, float thresh)
{
    int i;
    int s = 0;
    for (i = 0; i < net->n; ++i) {
        layer l = net->layers[i];
        if (l.type == YOLO) {
            s += yolo_num_detections(l, thresh);
        }
        if (l.type == YOLO4) {
            s += yolo4_num_detections(l, thresh);
        }
        if (l.type == DETECTION || l.type == REGION) {
            s += l.w*l.h*l.n;
        }
    }
    return s;
}

int num_detections_batch_y4(network *net, float thresh, int batch)
{
    int i;
    int s = 0;
    for (i = 0; i < net->n; ++i) {
        layer l = net->layers[i];
        if (l.type == YOLO4) {
            s += yolo4_num_detections_batch(l, thresh, batch);
        }
        if (l.type == DETECTION || l.type == REGION || l.type == YOLO) {
            s += l.w*l.h*l.n;
        }
    }
    return s;
}

detection *make_network_boxes_y4(network *net, float thresh, int *num)
{
    int i;
    layer l = net->layers[net->n - 1];
    for (i = 0; i < net->n; ++i) {
        layer l_tmp = net->layers[i];
        if (l_tmp.type == YOLO || l_tmp.type == YOLO4 || l_tmp.type == DETECTION || l_tmp.type == REGION) {
            l = l_tmp;
            break;
        }
    }

    int nboxes = num_detections_y4(net, thresh);
    if (num) *num = nboxes;
    detection* dets = (detection*)calloc(nboxes, sizeof(detection));
    for (i = 0; i < nboxes; ++i) {
        dets[i].prob = (float*)calloc(l.classes, sizeof(float));
        // tx,ty,tw,th uncertainty
        dets[i].uc = NULL;

        if (l.coords > 4) dets[i].mask = (float*)calloc(l.coords - 4, sizeof(float));
        else dets[i].mask = NULL;

        if(l.embedding_output) dets[i].embeddings = (float*)calloc(l.embedding_size, sizeof(float));
        else dets[i].embeddings = NULL;
        dets[i].embedding_size = l.embedding_size;
    }
    return dets;
}

detection *make_network_boxes_batch_y4(network *net, float thresh, int *num, int batch)
{
    int i;
    layer l = net->layers[net->n - 1];
    for (i = 0; i < net->n; ++i) {
        layer l_tmp = net->layers[i];
        if (l_tmp.type == YOLO || l_tmp.type == YOLO4 || l_tmp.type == DETECTION || l_tmp.type == REGION) {
            l = l_tmp;
            break;
        }
    }

    int nboxes = num_detections_batch_y4(net, thresh, batch);
    assert(num != NULL);
    *num = nboxes;
    detection* dets = (detection*)calloc(nboxes, sizeof(detection));
    for (i = 0; i < nboxes; ++i) {
        dets[i].prob = (float*)calloc(l.classes, sizeof(float));
        // tx,ty,tw,th uncertainty
        dets[i].uc = NULL;

        if (l.coords > 4) dets[i].mask = (float*)calloc(l.coords - 4, sizeof(float));
        else dets[i].mask = NULL;

        if (l.embedding_output) dets[i].embeddings = (float*)calloc(l.embedding_size, sizeof(float));
        else dets[i].embeddings = NULL;
        dets[i].embedding_size = l.embedding_size;
    }
    return dets;
}

void custom_get_region_detections(layer l, int w, int h, int net_w, int net_h, float thresh, int *map, float hier, int relative, detection *dets, int letter)
{
    box* boxes = (box*)calloc(l.w * l.h * l.n, sizeof(box));
    float** probs = (float**)calloc(l.w * l.h * l.n, sizeof(float*));
    int i, j;
    for (j = 0; j < l.w*l.h*l.n; ++j) probs[j] = (float*)calloc(l.classes, sizeof(float));
    get_region_boxes(l, 1, 1, thresh, probs, boxes, 0, map);
    for (j = 0; j < l.w*l.h*l.n; ++j) {
        dets[j].classes = l.classes;
        dets[j].bbox = boxes[j];
        dets[j].objectness = 1;
        for (i = 0; i < l.classes; ++i) {
            dets[j].prob[i] = probs[j][i];
        }
    }

    free(boxes);
    free_ptrs((void **)probs, l.w*l.h*l.n);

    //correct_region_boxes(dets, l.w*l.h*l.n, w, h, net_w, net_h, relative);
    correct_yolo4_boxes(dets, l.w*l.h*l.n, w, h, net_w, net_h, relative, letter);
}

void fill_network_boxes_y4(network *net, int w, int h, float thresh, float hier, int *map, int relative, detection *dets, int letter)
{
    int prev_classes = -1;
    int j;
    for (j = 0; j < net->n; ++j) {
        layer l = net->layers[j];
        if (l.type == YOLO || l.type == YOLO4) {
            int count = get_yolo4_detections(l, w, h, net->w, net->h, thresh, map, relative, dets, letter);
            dets += count;
            if (prev_classes < 0) prev_classes = l.classes;
            else if (prev_classes != l.classes) {
                printf(" Error: Different [yolo] layers have different number of classes = %d and %d - check your cfg-file! \n",
                       prev_classes, l.classes);
            }
        }
        if (l.type == REGION) {
            custom_get_region_detections(l, w, h, net->w, net->h, thresh, map, hier, relative, dets, letter);
            //get_region_detections(l, w, h, net->w, net->h, thresh, map, hier, relative, dets);
            dets += l.w*l.h*l.n;
        }
        if (l.type == DETECTION) {
            get_detection_detections(l, w, h, thresh, dets);
            dets += l.w*l.h*l.n;
        }
    }
}

void fill_network_boxes_batch_y4(network *net, int w, int h, float thresh, float hier, int *map, int relative, detection *dets, int letter, int batch)
{
    int prev_classes = -1;
    int j;
    for (j = 0; j < net->n; ++j) {
        layer l = net->layers[j];
        if (l.type == YOLO || l.type == YOLO4) {
            int count = get_yolo4_detections_batch(l, w, h, net->w, net->h, thresh, map, relative, dets, letter, batch);
            dets += count;
            if (prev_classes < 0) prev_classes = l.classes;
            else if (prev_classes != l.classes) {
                printf(" Error: Different [yolo] layers have different number of classes = %d and %d - check your cfg-file! \n",
                       prev_classes, l.classes);
            }
        }
        if (l.type == REGION) {
            custom_get_region_detections(l, w, h, net->w, net->h, thresh, map, hier, relative, dets, letter);
            //get_region_detections(l, w, h, net->w, net->h, thresh, map, hier, relative, dets);
            dets += l.w*l.h*l.n;
        }
        if (l.type == DETECTION) {
            get_detection_detections(l, w, h, thresh, dets);
            dets += l.w*l.h*l.n;
        }
    }
}

detection *get_network_boxes(network *net, int w, int h, float thresh, float hier, int *map, int relative, int *num)
{
	detection *dets = make_network_boxes(net, thresh, num);
	fill_network_boxes(net, w, h, thresh, hier, map, relative, dets);
	return dets;
}

detection *get_network_boxes_y4(network *net, int w, int h, float thresh, float hier, int *map, int relative, int *num, int letter)
{
    detection *dets = make_network_boxes_y4(net, thresh, num);
    fill_network_boxes_y4(net, w, h, thresh, hier, map, relative, dets, letter);
    return dets;
}

void free_detections(detection *dets, int n)
{
	int i;
	for(i = 0; i < n; ++i){
		free(dets[i].prob);
		if(dets[i].mask) free(dets[i].mask);
	}
	free(dets);
}

float *network_predict_image(network *net, image im)
{
	image imr = letterbox_image(im, net->w, net->h);
	set_batch_network(net, 1);
	float *p = network_predict(net, imr.data);
	free_image(imr);
	return p;
}

int network_width(network *net){return net->w;}
int network_height(network *net){return net->h;}

matrix network_predict_data_multi(network *net, data test, int n)
{
	int i,j,b,m;
	int k = net->outputs;
	matrix pred = make_matrix(test.X.rows, k);
	float *X = (float*)calloc(net->batch*test.X.rows, sizeof(float));
	for(i = 0; i < test.X.rows; i += net->batch){
		for(b = 0; b < net->batch; ++b){
			if(i+b == test.X.rows) break;
			memcpy(X+b*test.X.cols, test.X.vals[i+b], test.X.cols*sizeof(float));
		}
		for(m = 0; m < n; ++m){
			float *out = network_predict(net, X);
			for(b = 0; b < net->batch; ++b){
				if(i+b == test.X.rows) break;
				for(j = 0; j < k; ++j){
					pred.vals[i+b][j] += out[j+b*k]/n;
				}
			}
		}
	}
	free(X);
	return pred;
}

matrix network_predict_data(network *net, data test)
{
	int i,j,b;
	int k = net->outputs;
	matrix pred = make_matrix(test.X.rows, k);
	float *X = (float*)calloc(net->batch*test.X.cols, sizeof(float));
	for(i = 0; i < test.X.rows; i += net->batch){
		for(b = 0; b < net->batch; ++b){
			if(i+b == test.X.rows) break;
			memcpy(X+b*test.X.cols, test.X.vals[i+b], test.X.cols*sizeof(float));
		}
		float *out = network_predict(net, X);
		for(b = 0; b < net->batch; ++b){
			if(i+b == test.X.rows) break;
			for(j = 0; j < k; ++j){
				pred.vals[i+b][j] = out[j+b*k];
			}
		}
	}
	free(X);
	return pred;
}

void print_network(network *net)
{
	int i,j;
	for(i = 0; i < net->n; ++i){
		layer l = net->layers[i];
		float *output = l.output;
		int n = l.outputs;
		float mean = mean_array(output, n);
		float vari = variance_array(output, n);
		fprintf(stderr, "Layer %d - Mean: %f, Variance: %f\n",i,mean, vari);
		if(n > 100) n = 100;
		for(j = 0; j < n; ++j) fprintf(stderr, "%f, ", output[j]);
		if(n == 100)fprintf(stderr,".....\n");
		fprintf(stderr, "\n");
	}
}

void compare_networks(network *n1, network *n2, data test)
{
	matrix g1 = network_predict_data(n1, test);
	matrix g2 = network_predict_data(n2, test);
	int i;
	int a,b,c,d;
	a = b = c = d = 0;
	for(i = 0; i < g1.rows; ++i){
		int truth = max_index(test.y.vals[i], test.y.cols);
		int p1 = max_index(g1.vals[i], g1.cols);
		int p2 = max_index(g2.vals[i], g2.cols);
		if(p1 == truth){
			if(p2 == truth) ++d;
			else ++c;
		}else{
			if(p2 == truth) ++b;
			else ++a;
		}
	}
	printf("%5d %5d\n%5d %5d\n", a, b, c, d);
	float num = pow((abs(b - c) - 1.), 2.);
	float den = b + c;
	printf("%f\n", num/den);
}

float network_accuracy(network *net, data d)
{
	matrix guess = network_predict_data(net, d);
	float acc = matrix_topk_accuracy(d.y, guess,1);
	free_matrix(guess);
	return acc;
}

float *network_accuracies(network *net, data d, int n)
{
	static float acc[2];
	matrix guess = network_predict_data(net, d);
	acc[0] = matrix_topk_accuracy(d.y, guess, 1);
	acc[1] = matrix_topk_accuracy(d.y, guess, n);
	free_matrix(guess);
	return acc;
}

layer get_network_output_layer_cgan(network *net)
{
    int i, c;
    for(i = net->n-1, c = 0; i >= 0; --i){
        if(c++ > 0 && net->layers[i].type == CONVOLUTIONAL) break;
    }
    return net->layers[i];
}

layer get_network_output_layer(network *net)
{
	int i;
	for(i = net->n - 1; i >= 0; --i){
		if(net->layers[i].type != COST) break;
	}
	return net->layers[i];
}

float network_accuracy_multi(network *net, data d, int n)
{
	matrix guess = network_predict_data_multi(net, d, n);
	float acc = matrix_topk_accuracy(d.y, guess,1);
	free_matrix(guess);
	return acc;
}

void free_network(network *net)
{
	int i;
	for(i = 0; i < net->n; ++i){
		free_layer(net->layers[i]);
	}
	free(net->layers);
#ifdef GPU
	if (gpu_index >= 0) {
		if (net->input_gpu.ptr) opencl_free(net->input_gpu);
		if (net->truth_gpu.ptr) opencl_free(net->truth_gpu);
		if (net->delta_gpu.ptr) opencl_free(net->delta_gpu);
		if (net->workspace_gpu.ptr) opencl_free(net->workspace_gpu);
	}
	else {
        if (net->input) free(net->input);
        if (net->truth) free(net->truth);
        if (net->delta) free(net->delta);
		free(net->workspace);
	}
#else
    if (net->input) free(net->input);
	if (net->truth) free(net->truth);
	if (net->delta) free(net->delta);
	free(net->workspace);
#endif
    free(net->seen);
    free(net->t);
    free(net->cost);
    free(net->cur_iteration);
    free(net->total_bbox);
    free(net->rewritten_bbox);
	free(net);
}

// Some day...
// ^ What the hell is this comment for?

layer network_output_layer(network *net)
{
	int i;
	for(i = net->n - 1; i >= 0; --i){
		if(net->layers[i].type != COST) break;
	}
	return net->layers[i];
}

int network_inputs(network *net)
{
	return net->layers[0].inputs;
}

int network_outputs(network *net)
{
	return network_output_layer(net).outputs;
}

float *network_output(network *net)
{
	return network_output_layer(net).output;
}

#ifdef GPU

void forward_network_gpu(network *netp)
{
	network net = *netp;
	opencl_push_array(net.input_gpu, net.input, net.inputs*net.batch);
	if(net.train && net.truth){
		opencl_push_array(net.truth_gpu, net.truth, net.truths*net.batch);
	}
	int i;
	for(i = 0; i < net.n; ++i){
		net.index = i;
		layer l = net.layers[i];
		if(net.train && l.delta){
			fill_gpu(l.outputs * l.batch, 0, l.delta_gpu, 1);
		}
#ifdef BENCHMARK
		clock_t t;
		t = clock();
#endif
		l.forward_gpu(l, net);
#ifdef BENCHMARK
		t = clock() - t;
		double time_taken = ((double)t);
		const char* layerName[] = { "CONVOLUTIONAL","DECONVOLUTIONAL","CONNECTED", "MAXPOOL", "SOFTMAX", "DETECTION", "DROPOUT", "CROP", "ROUTE", "COST", "NORMALIZATION", "AVGPOOL", "LOCAL", "SHORTCUT", "ACTIVE", "RNN", "GRU", "LSTM", "CRNN", "BATCHNORM", "NETWORK", "XNOR", "REGION", "YOLO", "YOLO4", "ISEG", "REORG", "UPSAMPLE", "LOGXENT", "L2NORM", "BLANK"};
		printf("FW %s\t%d\n", layerName[(int)l.type], (int)time_taken);
#endif
        net.input = l.output;
        net.input_gpu = l.output_gpu;
		if(l.truth) {
			net.truth_gpu = l.output_gpu;
			net.truth = l.output;
		}
	}

	// clFlush(opencl_queues[opencl_device_id_t]);

	pull_network_output(netp);
	if(net.train) calc_network_cost(netp);
}

void backward_network_gpu(network *netp)
{
	int i;
	network net = *netp;
	network orig = net;
	for(i = net.n-1; i >= 0; --i){
		layer l = net.layers[i];
		if(l.stopbackward) break;
        if(i == 0){
			net = orig;
		}else{
			layer prev = net.layers[i-1];
			net.delta = prev.delta;
            net.delta_gpu = prev.delta_gpu;
            net.input = prev.output;
			net.input_gpu = prev.output_gpu;
		}
		net.index = i;
#ifdef BENCHMARK
		clock_t t;
		t = clock();
#endif
		l.backward_gpu(l, net);
#ifdef BENCHMARK
		t = clock() - t;
		double time_taken = ((double)t);
		const char* layerName[] = { "CONVOLUTIONAL","DECONVOLUTIONAL","CONNECTED", "MAXPOOL", "SOFTMAX", "DETECTION", "DROPOUT", "CROP", "ROUTE", "COST", "NORMALIZATION", "AVGPOOL", "LOCAL", "SHORTCUT", "ACTIVE", "RNN", "GRU", "LSTM", "CRNN", "BATCHNORM", "NETWORK", "XNOR", "REGION", "YOLO", "YOLO4", "ISEG", "REORG", "UPSAMPLE", "LOGXENT", "L2NORM", "BLANK"};
		printf("BW %s\t%d\n", layerName[(int)l.type], (int)time_taken);
#endif
	}

        // clFlush(opencl_queues[opencl_device_id_t]);
}

void update_network_gpu(network *netp)
{
	network net = *netp;
	int i;
	update_args a = {0};
	a.batch = net.batch*net.subdivisions;
	a.learning_rate = get_current_rate(netp);
	a.momentum = net.momentum;
	a.decay = net.decay;
	a.adam = net.adam;
	a.B1 = net.B1;
	a.B2 = net.B2;
	a.eps = net.eps;
	++*net.t;
	a.t = (*net.t);

	for(i = 0; i < net.n; ++i){
		layer l = net.layers[i];
        if(l.update_gpu){
#ifdef BENCHMARK
			clock_t t;
			t = clock();
#endif
			l.update_gpu(l, a);
#ifdef BENCHMARK
			t = clock() - t;
			double time_taken = ((double)t);
			const char* layerName[] = { "CONVOLUTIONAL","DECONVOLUTIONAL","CONNECTED", "MAXPOOL", "SOFTMAX", "DETECTION", "DROPOUT", "CROP", "ROUTE", "COST", "NORMALIZATION", "AVGPOOL", "LOCAL", "SHORTCUT", "ACTIVE", "RNN", "GRU", "LSTM", "CRNN", "BATCHNORM", "NETWORK", "XNOR", "REGION", "YOLO", "YOLO4", "ISEG", "REORG", "UPSAMPLE", "LOGXENT", "L2NORM", "BLANK"};
			printf("UP %s\t%d\n", layerName[(int)l.type], (int)time_taken);
#endif
		}
	}

        // clFlush(opencl_queues[opencl_device_id_t]);
}

void harmless_update_network_gpu(network *netp)
{
	network net = *netp;
	opencl_set_device(net.gpu_index);
	int i;
	for(i = 0; i < net.n; ++i){
		layer l = net.layers[i];
		if(l.weight_updates_gpu.ptr) fill_gpu(l.nweights, 0, l.weight_updates_gpu, 1);
		if(l.bias_updates_gpu.ptr) fill_gpu(l.nbiases, 0, l.bias_updates_gpu, 1);
		if(l.scale_updates_gpu.ptr) fill_gpu(l.nbiases, 0, l.scale_updates_gpu, 1);
	}
}

typedef struct {
	network *net;
	data d;
	data o;
	float *err;
} train_args;

void *train_thread(void *ptr)
{
	train_args args = *(train_args*)ptr;
	free(ptr);
	opencl_set_device(args.net->gpu_index);
	*args.err = train_network(args.net, args.d);
	return 0;
}

void *train_thread_cgan(void *ptr)
{
    train_args args = *(train_args*)ptr;
    free(ptr);
    opencl_set_device(args.net->gpu_index);
    *args.err = train_network_cgan(args.net, args.d, args.o);
    return 0;
}

pthread_t train_network_in_thread(network *net, data d, float *err)
{
	pthread_t thread;
	train_args *ptr = (train_args *)calloc(1, sizeof(train_args));
	ptr->net = net;
	ptr->d = d;
	ptr->err = err;
	if(pthread_create(&thread, 0, train_thread, ptr)) error("Thread creation failed");
	return thread;
}
pthread_t train_network_in_thread_cgan(network *net, data d, data o, float *err)
{
    pthread_t thread;
    train_args *ptr = (train_args *)calloc(1, sizeof(train_args));
    ptr->net = net;
    ptr->d = d;
    ptr->o = o;
    ptr->err = err;
    if(pthread_create(&thread, 0, train_thread_cgan, ptr)) error("Thread creation failed");
    return thread;
}

void merge_weights(layer l, layer base)
{
	if (l.type == CONVOLUTIONAL) {
		axpy_cpu(l.n, 1, l.bias_updates, 1, base.biases, 1);
		axpy_cpu(l.nweights, 1, l.weight_updates, 1, base.weights, 1);
		if (l.scales) {
			axpy_cpu(l.n, 1, l.scale_updates, 1, base.scales, 1);
		}
	} else if(l.type == CONNECTED) {
		axpy_cpu(l.outputs, 1, l.bias_updates, 1, base.biases, 1);
		axpy_cpu(l.outputs*l.inputs, 1, l.weight_updates, 1, base.weights, 1);
	}
}

void scale_weights(layer l, float s)
{
    if (l.type == CONVOLUTIONAL) {
        scal_cpu(l.n, s, l.biases, 1);
        scal_cpu(l.nweights, s, l.weights, 1);
        if (l.scales) {
            scal_cpu(l.n, s, l.scales, 1);
        }
    } else if(l.type == CONNECTED) {
        scal_cpu(l.outputs, s, l.biases, 1);
        scal_cpu(l.outputs*l.inputs, s, l.weights, 1);
    }
}

void pull_weights(layer l)
{
    if(l.type == CONVOLUTIONAL || l.type == DECONVOLUTIONAL){
        opencl_pull_array_map(l.biases_gpu, l.bias_updates, l.n);
        opencl_pull_array_map(l.weights_gpu, l.weight_updates, l.nweights);
        if(l.scales) opencl_pull_array_map(l.scales_gpu, l.scale_updates, l.n);
    } else if(l.type == CONNECTED){
        opencl_pull_array_map(l.biases_gpu, l.bias_updates, l.outputs);
        opencl_pull_array_map(l.weights_gpu, l.weight_updates, l.outputs*l.inputs);
    }
}

void push_weights(layer l)
{
    if(l.type == CONVOLUTIONAL || l.type == DECONVOLUTIONAL){
        opencl_push_array_map(l.biases_gpu, l.bias_updates, l.n);
        opencl_push_array_map(l.weights_gpu, l.weight_updates, l.nweights);
        if(l.scales) opencl_push_array_map(l.scales_gpu, l.scale_updates, l.n);
    } else if(l.type == CONNECTED){
        opencl_push_array_map(l.biases_gpu, l.bias_updates, l.outputs);
        opencl_push_array_map(l.weights_gpu, l.weight_updates, l.outputs*l.inputs);
    }
}

void distribute_weights(layer l, layer base)
{
	if (l.type == CONVOLUTIONAL || l.type == DECONVOLUTIONAL) {
		opencl_push_array(l.biases_gpu, base.biases, l.n);
		opencl_push_array(l.weights_gpu, base.weights, l.nweights);
		if (base.scales) opencl_push_array(l.scales_gpu, base.scales, l.n);
	} else if (l.type == CONNECTED) {
		opencl_push_array(l.biases_gpu, base.biases, l.outputs);
		opencl_push_array(l.weights_gpu, base.weights, l.outputs*l.inputs);
	}
}


void sync_layer(network **nets, int n, int j)
{
	int i;
	network *net = nets[0];
	layer base = net->layers[j];
	scale_weights(base, 0.0f);
	for (i = 0; i < n; ++i) {
		opencl_set_device(nets[i]->gpu_index);
		layer l = nets[i]->layers[j];
		pull_weights(l);
		merge_weights(l, base);
	}
	scale_weights(base, 1.0f/n);
	for (i = 0; i < n; ++i) {
		opencl_set_device(nets[i]->gpu_index);
		layer l = nets[i]->layers[j];
        push_weights(l);
	}
}

typedef struct{
    network **nets;
    int n;
    int j;
} sync_args;

void *sync_layer_thread(void *ptr)
{
    sync_args args = *(sync_args*)ptr;
    sync_layer(args.nets, args.n, args.j);
    free(ptr);
    return 0;
}

pthread_t sync_layer_in_thread(network **nets, int n, int j)
{
    pthread_t thread;
    sync_args *ptr = (sync_args *)calloc(1, sizeof(sync_args));
    ptr->nets = nets;
    ptr->n = n;
    ptr->j = j;
    if(pthread_create(&thread, 0, sync_layer_thread, ptr)) error("Thread creation failed");
    return thread;
}

void sync_nets(network **nets, int n, int interval)
{
    int j;
    int layers = nets[0]->n;
    pthread_t *threads = (pthread_t *) calloc(layers, sizeof(pthread_t));

    *(nets[0]->seen) += interval * (n-1) * nets[0]->batch * nets[0]->subdivisions;
    for (j = 0; j < n; ++j){
        *(nets[j]->seen) = *(nets[0]->seen);
    }
    for (j = 0; j < layers; ++j) {
        threads[j] = sync_layer_in_thread(nets, n, j);
    }
    for (j = 0; j < layers; ++j) {
        pthread_join(threads[j], 0);
    }
    free(threads);
}

float train_networks(network **nets, int n, data d, int interval)
{
	int i;
	int batch = nets[0]->batch;
	int subdivisions = nets[0]->subdivisions;
	assert(batch * subdivisions * n == d.X.rows);
	pthread_t *threads = (pthread_t *) calloc(n, sizeof(pthread_t));
	float *errors = (float *) calloc(n, sizeof(float));

	float sum = 0;
	for(i = 0; i < n; ++i){
		data p = get_data_part(d, i, n);
		threads[i] = train_network_in_thread(nets[i], p, errors + i);
	}
	for(i = 0; i < n; ++i){
		pthread_join(threads[i], 0);
		//printf("%f\n", errors[i]);
		sum += errors[i];
	}
	if (get_current_batch(nets[0]) % interval == 0) {
		printf("Syncing... ");
		fflush(stdout);
		sync_nets(nets, n, interval);
		printf("Done!\n");
	}
	free(threads);
	free(errors);
	return (float)sum/(n);
}

float train_networks_cgan(network **nets, int n, data d, data o, int interval)
{
    int i;
    int batch = nets[0]->batch;
    int subdivisions = nets[0]->subdivisions;
    assert(batch * subdivisions * n == d.X.rows);
    assert(batch * subdivisions * n == o.X.rows);
    pthread_t *threads = (pthread_t *) calloc(n, sizeof(pthread_t));
    float *errors = (float *) calloc(n, sizeof(float));

    float sum = 0;
    for(i = 0; i < n; ++i){
        data p = get_data_part(d, i, n);
        data r = get_data_part(o, i, n);
        threads[i] = train_network_in_thread_cgan(nets[i], p, r, errors + i);
    }
    for(i = 0; i < n; ++i){
        pthread_join(threads[i], 0);
        //printf("%f\n", errors[i]);
        sum += errors[i];
    }
    if (get_current_batch(nets[0]) % interval == 0) {
        printf("Syncing... ");
        fflush(stdout);
        sync_nets(nets, n, interval);
        printf("Done!\n");
    }
    free(threads);
    free(errors);
    return (float)sum/(n);
}

void pull_network_output(network *net)
{
    layer l = get_network_output_layer(net);
    opencl_pull_array(l.output_gpu, l.output, l.outputs*l.batch);
}

#endif
