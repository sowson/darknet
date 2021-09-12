#include <math.h>
#include "darknet.h"

void probe_cgan(network *net, char *filepath, char *name);
void ideal_cgan(network *net, char *filepath, char *name);

void train_cgan(char *datacfg, char *gcfg, char *gweight, int *gpus, int ngpus, int clear)
{
    int i;

    float avg_loss = -1;
    char *base = basecfg(gcfg);
    printf("%s\n", base);
    printf("%d\n", ngpus);
    network **nets = (network**)calloc(ngpus, sizeof(network*));

    for(i = 0; i < ngpus; ++i){
#ifdef GPU
        if(gpu_index >= 0){
            opencl_set_device(i);
        }
#endif
        nets[i] = load_network(gcfg, gweight, clear);
        nets[i]->learning_rate *= ngpus;
#ifdef GPU
        opencl_free(nets[i]->delta_gpu);
#else
        free(nets[i]->delta);
#endif
        nets[i]->delta = 0;
    }

    network *net = nets[0];

    int imgs = net->batch * net->subdivisions * ngpus;

#ifndef BENCHMARK
    printf("Learning Rate: %g, Momentum: %g, Decay: %g\n", net->learning_rate, net->momentum, net->decay);
#endif
    list *options = read_data_cfg(datacfg);

    char *backup_directory = option_find_str(options, "backup", "/backup/");
    int tag = option_find_int_quiet(options, "tag", 0);
    char *label_list = option_find_str(options, "labels", "data/labels.list");
    char *train_list = option_find_str(options, "trainA", "data/trainA.list");
    char *trrev_list = option_find_str(options, "trainB", "data/trainB.list");
    char *tree = option_find_str(options, "tree", 0);
    if (tree) net->hierarchy = read_tree(tree);
    int classes = option_find_int(options, "classes", 2);

    char **labels = 0;
    if(!tag){
        labels = get_labels(label_list);
    }
    list *plist = get_paths(train_list);
    char **paths = (char **)list_to_array(plist);
    printf("%d\n", plist->size);
    int N = plist->size;

    list *rplist = get_paths(trrev_list);
    char **rpaths = (char **)list_to_array(rplist);
    
    double time;

    load_args args = {0};
    args.w = net->w;
    args.h = net->h;
    args.threads = 32;
    args.hierarchy = net->hierarchy;

    args.min = net->min_ratio*net->w;
    args.max = net->max_ratio*net->w;
    printf("%d %d\n", args.min, args.max);
    args.angle = net->angle;
    args.aspect = net->aspect;
    args.exposure = net->exposure;
    args.saturation = net->saturation;
    args.hue = net->hue;
    args.size = net->w;

    args.paths = paths;
    args.classes = classes;
    args.n = imgs;
    args.m = N;
    args.labels = labels;
    if (tag){
        args.type = TAG_DATA;
    } else {
        args.type = CLASSIFICATION_DATA;
    }

    load_args rargs = {0};
    rargs.w = net->w;
    rargs.h = net->h;
    rargs.threads = 32;
    rargs.hierarchy = net->hierarchy;

    rargs.min = net->min_ratio*net->w;
    rargs.max = net->max_ratio*net->w;
    printf("%d %d\n", rargs.min, rargs.max);
    rargs.angle = net->angle;
    rargs.aspect = net->aspect;
    rargs.exposure = net->exposure;
    rargs.saturation = net->saturation;
    rargs.hue = net->hue;
    rargs.size = net->w;

    rargs.paths = rpaths;
    rargs.classes = classes;
    rargs.n = imgs;
    rargs.m = N;
    rargs.labels = labels;
    rargs.type = args.type;
    
    data train, trrev;
    data buffer, rbuffer;
    pthread_t load_thread;
    pthread_t load_rthread;
    args.d = &buffer;
    rargs.d = &rbuffer;
    load_thread = load_data(args);
    load_rthread = load_data(rargs);

    int count = 0;
    int epoch = (*net->seen)/N;

    if(count == 0) {
        char buff[256];
        sprintf(buff, "%s/%s.start.conv.weights",backup_directory,base);
        save_weights(net, buff);
    }

#ifdef LOSS_ONLY
    time = what_time_is_it_now();
#endif

    while(get_current_batch(net) < net->max_batches || net->max_batches == 0){
        if(net->random && count++%40 == 0){
#if !defined(BENCHMARK) && !defined(LOSS_ONLY)
            printf("Resizing\n");
#endif
            int dim = (rand() % 11 + 4) * 32;
            //if (get_current_batch(net)+200 > net->max_batches) dim = 608;
            //int dim = (rand() % 4 + 16) * 32;
#if !defined(BENCHMARK) && !defined(LOSS_ONLY)
            printf("%d\n", dim);
#endif
            args.w = dim;
            args.h = dim;
            args.size = dim;
            args.min = net->min_ratio*dim;
            args.max = net->max_ratio*dim;
#ifndef BENCHMARK
            printf("%d %d\n", args.min, args.max);
#endif
            pthread_join(load_thread, 0);
            train = buffer;
            free_data(buffer);
            pthread_join(load_rthread, 0);
            trrev = rbuffer;
            free_data(rbuffer);
            load_thread = load_data(args);
            load_rthread = load_data(rargs);

            for(i = 0; i < ngpus; ++i){
                resize_network(nets[i], dim, dim);
            }
            net = nets[0];
        }
#ifndef LOSS_ONLY
        time = what_time_is_it_now();
#endif
        pthread_join(load_thread, 0);
        train = copy_data(buffer);
        free_data(buffer);
        pthread_join(load_rthread, 0);
        trrev = copy_data(rbuffer);
        free_data(rbuffer);
        load_thread = load_data(args);
        load_rthread = load_data(rargs);
#ifndef LOSS_ONLY
        printf("Loaded: %lf seconds\n", what_time_is_it_now()-time);
#endif
#ifndef LOSS_ONLY
        time = what_time_is_it_now();
#endif
        float loss = 0;
#ifdef GPU
        if (gpu_index >= 0) {
            if (ngpus == 1) {
                loss = train_network_cgan(net, train, trrev);
            } else {
                loss = train_networks_cgan(nets, ngpus, train, trrev, 4);
            }
        }
        else {
            loss = train_network(net, train);
        }
#else
        loss = train_network(net, train);
#endif
        if(avg_loss == -1) avg_loss = loss;
        avg_loss = avg_loss*.9 + loss*.1;
#ifdef LOSS_ONLY
        printf("%lf\t%f\n", what_time_is_it_now()-time, loss);
#else
        printf("%ld, %.3f: %f, %f avg, %f rate, %lf seconds, %ld images\n", get_current_batch(net), (float)(*net->seen)/N, loss, avg_loss, get_current_rate(net), what_time_is_it_now()-time, *net->seen);
#endif
        free_data(train);
        free_data(trrev);
        if(*net->seen/N > epoch){
            epoch = *net->seen/N;
            char buff[256];
            sprintf(buff, "%s/%s_%d.weights",backup_directory,base, epoch);
            save_weights(net, buff);
        }
        if(get_current_batch(net)%1000 == 0){
            char buff[256];
            sprintf(buff, "%s/%s.backup", backup_directory, base);
            save_weights(net, buff);
        }
#ifdef OPENCV_CGAN
        if(get_current_batch(net)%32 == 0){
            cvNamedWindow("ifrog", CV_WINDOW_NORMAL);
            cvMoveWindow("ifrog", 1*256+64, 64);
            ideal_cgan(net, "/Users/piotr/cgan/data/ccifar/test/7_frog.png", "ifrog");
            cvNamedWindow("frog", CV_WINDOW_NORMAL);
            cvMoveWindow("frog", 1*256+64, 256+64+32);
            probe_cgan(net, "/Users/piotr/cgan/data/gcifar/test/7_frog.png", "frog");
            cvNamedWindow("gfrog", CV_WINDOW_NORMAL);
            cvMoveWindow("gfrog", 1*256+64, 256+256+64+64);
            ideal_cgan(net, "/Users/piotr/cgan/data/gcifar/test/7_frog.png", "gfrog");

            cvNamedWindow("iship", CV_WINDOW_NORMAL);
            cvMoveWindow("iship", 2*256+64, 64);
            ideal_cgan(net, "/Users/piotr/cgan/data/ccifar/test/15_ship.png", "iship");
            cvNamedWindow("ship", CV_WINDOW_NORMAL);
            cvMoveWindow("ship", 2*256+64, 256+64+32);
            probe_cgan(net, "/Users/piotr/cgan/data/gcifar/test/15_ship.png", "ship");
            cvNamedWindow("gship", CV_WINDOW_NORMAL);
            cvMoveWindow("gship", 2*256+64, 256+256+64+64);
            ideal_cgan(net, "/Users/piotr/cgan/data/gcifar/test/15_ship.png", "gship");

            cvNamedWindow("iairplane", CV_WINDOW_NORMAL);
            cvMoveWindow("iairplane", 3*256+64, 64);
            ideal_cgan(net, "/Users/piotr/cgan/data/ccifar/test/27_airplane.png", "iairplane");
            cvNamedWindow("airplane", CV_WINDOW_NORMAL);
            cvMoveWindow("airplane", 3*256+64, 256+64+32);
            probe_cgan(net, "/Users/piotr/cgan/data/gcifar/test/27_airplane.png", "airplane");
            cvNamedWindow("gairplane", CV_WINDOW_NORMAL);
            cvMoveWindow("gairplane", 3*256+64, 256+256+64+64);
            ideal_cgan(net, "/Users/piotr/cgan/data/gcifar/test/27_airplane.png", "gairplane");

            cvNamedWindow("icat", CV_WINDOW_NORMAL);
            cvMoveWindow("icat", 4*256+64, 64);
            ideal_cgan(net, "/Users/piotr/cgan/data/ccifar/test/46_cat.png", "icat");
            cvNamedWindow("cat", CV_WINDOW_NORMAL);
            cvMoveWindow("cat", 4*256+64, 256+64+32);
            probe_cgan(net, "/Users/piotr/cgan/data/gcifar/test/46_cat.png", "cat");
            cvNamedWindow("gcat", CV_WINDOW_NORMAL);
            cvMoveWindow("gcat", 4*256+64, 256+256+64+64);
            ideal_cgan(net, "/Users/piotr/cgan/data/gcifar/test/46_cat.png", "gcat");

            cvNamedWindow("ihorse", CV_WINDOW_NORMAL);
            cvMoveWindow("ihorse", 5*256+64, 64);
            ideal_cgan(net, "/Users/piotr/cgan/data/ccifar/test/60_horse.png", "ihorse");
            cvNamedWindow("horse", CV_WINDOW_NORMAL);
            cvMoveWindow("horse", 5*256+64, 256+64+32);
            probe_cgan(net, "/Users/piotr/cgan/data/gcifar/test/60_horse.png", "horse");
            cvNamedWindow("ghorse", CV_WINDOW_NORMAL);
            cvMoveWindow("ghorse", 5*256+64, 256+256+64+64);
            ideal_cgan(net, "/Users/piotr/cgan/data/gcifar/test/60_horse.png", "ghorse");
        }
#endif
#ifdef GPU_STATS
        opencl_dump_mem_stat();
#endif
#ifdef BENCHMARK
        break;
#endif
    }
    char buff[256];
    sprintf(buff, "%s/%s.weights", backup_directory, base);
    save_weights(net, buff);
    pthread_join(load_thread, 0);

    free_network(net);
    if(labels) free_ptrs((void**)labels, classes);
    free_ptrs((void**)paths, plist->size);
    //free_list(plist);
    free(base);
}

void probe_cgan(network *net, char *filepath, char* name) {
    int j, imlayer = 0;
    for (j = net->n-1; j >= 0; --j) {
        if (net->layers[j].out_c == 3 && net->layers[j].type == CONVOLUTIONAL) {
            imlayer = j;
            break;
        }
    }
    image im = load_image_color(filepath, 0, 0);
    int resize = im.w != net->w || im.h != net->h;
    image resized = resize ? letterbox_image(im, net->w, net->h) : im;
    //double time=clock();
    //float *X = resized.data;
    //time=what_time_is_it_now();
    //network_predict(net, X);
    network_predict_image(net, resized);
    image out = get_network_image_layer(net, imlayer);
    //constrain_image(out);
    //save_image(out, "out");
    image resizedm = resize_min(out, 256);
    show_image(resizedm, name, 1);
    free_image(im);
    free_image(resized);
    free_image(resizedm);
}

void ideal_cgan(network *net, char *filepath, char* name) {
    image im = load_image_color(filepath, 0, 0);
    image resized = resize_min(im, 256);
    show_image(resized, name, 1);
    free_image(resized);
}

void test_cgan(char *cfg, char *weights, char *filename, int gray)
{
    network *net = load_network(cfg, weights, 0);
    set_batch_network(net, 1);

    clock_t time;
    char buff[256];
    char *input = buff;
    int j, imlayer = 0;

    for (j = net->n-1; j >= 0; --j) {
        if (net->layers[j].out_c == 3 && net->layers[j].type == CONVOLUTIONAL) {
            imlayer = j;
            break;
        }
    }

    while(1){
        if(filename){
            strncpy(input, filename, 256);
        }else{
            printf("Enter Image Path: ");
            fflush(stdout);
            input = fgets(input, 256, stdin);
            if(!input) return;
            strtok(input, "\n");
        }
        image im = load_image_color(input, 0, 0);
        image resized = resize_min(im, net->w);
        image crop = crop_image(resized, (resized.w - net->w)/2, (resized.h - net->h)/2, net->w, net->h);
        if(gray) grayscale_image_3c(crop);

        float *X = crop.data;
        time=clock();
        network_predict(net, X);
        image out = get_network_image_layer(net, imlayer);
        //yuv_to_rgb(out);
        //constrain_image(out);
        printf("%s: Predicted in %f seconds.\n", input, sec(clock()-time));
        save_image(out, "out");
        show_image(crop, "crop", 1);
        show_image(out, "out", 0);

        free_image(im);
        free_image(resized);
        free_image(crop);
        if (filename) break;
    }
}

void run_cgan(int argc, char **argv)
{
    if(argc < 4){
        fprintf(stderr, "usage: %s %s [train/test/valid] [cfg] [weights (optional)]\n", argv[0], argv[1]);
        return;
    }

    char *gpu_list = find_char_arg(argc, argv, "-gpus", 0);
    int ngpus;
    int *gpus = read_intlist(gpu_list, &ngpus, gpu_index);

    int clear = find_arg(argc, argv, "-clear");
    int display = find_arg(argc, argv, "-display");
    int batches = find_int_arg(argc, argv, "-b", 0);

    char *data = argv[3];
    char *cfg = argv[4];
    char *weights = (argc > 5) ? argv[5] : 0;
    char *filename = (argc > 6) ? argv[6] : 0;
    if(0==strcmp(argv[2], "train")) train_cgan(data, cfg, weights, gpus, ngpus, clear);
    else if(0==strcmp(argv[2], "test")) test_cgan(cfg, weights, filename,0);
}
