#include "darknet.h"
#include "image.h"

#ifdef OPENCV_C
void train_vid_rnn(char *cfgfile, char *weightfile)
{
    char *train_videos = "data/vid/train.txt";
    char *backup_directory = "/home/piotr/backup/";
    srand(time(0));
    char *base = basecfg(cfgfile);
    printf("%s\n", base);
    float avg_loss = -1;
    network net = *parse_network_cfg(cfgfile);
    if(weightfile){
        load_weights(&net, weightfile);
    }
    printf("Learning Rate: %g, Momentum: %g, Decay: %g\n", net.learning_rate, net.momentum, net.decay);
    int imgs = net.batch*net.subdivisions;
    int i = *net.seen/imgs;

    list *plist = get_paths(train_videos);
    int N = plist->size;
    char **paths = (char **)list_to_array(plist);
    clock_t time;
    int steps = net.time_steps;
    int batch = net.batch / net.time_steps;

    network extractor = *parse_network_cfg("cfg/extractor.cfg");
    load_weights(&extractor, "/home/piotr/trained/yolo-coco.conv");

    while(get_current_batch(&net) < net.max_batches){
        i += 1;
        time=clock();
        float_pair p = get_rnn_vid_dataextractor, paths, N, batch, steps);

        copy_cpu(net.inputs*net.batch, p.x, 1, net.input, 1);
        copy_cpu(net.truths*net.batch, p.y, 1, net.truth, 1);
        float loss = train_network_datum(&net) / (net.batch);


        free(p.x);
        if (avg_loss < 0) avg_loss = loss;
        avg_loss = avg_loss*.9 + loss*.1;

        fprintf(stderr, "%d: %f, %f avg, %f rate, %lf seconds\n", i, loss, avg_loss, get_current_rate(&net), sec(clock()-time));
        if(i%100==0){
            char buff[256];
            sprintf(buff, "%s/%s_%d.weights", backup_directory, base, i);
            save_weights(&net, buff);
        }
        if(i%10==0){
            char buff[256];
            sprintf(buff, "%s/%s.backup", backup_directory, base);
            save_weights(&net, buff);
        }
    }
    char buff[256];
    sprintf(buff, "%s/%s_final.weights", backup_directory, base);
    save_weights(&net, buff);
}

void generate_vid_rnn(char *cfgfile, char *weightfile, char *filename) {
    network extractor = *parse_network_cfg(cfgfile);
    load_weights(&extractor, weightfile);

    network net = *parse_network_cfg(cfgfile);
    if (weightfile) {
        load_weights(&net, weightfile);
    }
    set_batch_network(&extractor, 1);
    set_batch_network(&net, 1);

    int i;
    CvCapture *cap = cvCaptureFromFile(filename);
    float *feat;
    float *next;
    image last;
    for (i = 0; i < 25; ++i) {
        image im = get_image_from_stream_cv(cap);
        image re = resize_image(im, extractor.w, extractor.h);
        feat = network_predict(&extractor, re.data);
        if (i > 0) {
            printf("%f %f\n", mean_array(feat, 14 * 14 * 512), variance_array(feat, 14 * 14 * 512));
            printf("%f %f\n", mean_array(next, 14 * 14 * 512), variance_array(next, 14 * 14 * 512));
            printf("%f\n", mse_arrayfeat, 14 * 14 * 512));
            axpy_cpu14 * 14 * 512, -1, feat, 1, next, 1);
            printf("%f\n", mse_arraynext, 14 * 14 * 512));
        }
        next = network_predict(&net, feat);

        free_image(im);

        free_image(save_reconstruction(extractor, 0, feat, "feat", i));
        free_image(save_reconstruction(extractor, 0, next, "next", i));

        if (i == 24) last = cv_copy_image(re);
        free_image(re);
    }
    for (i = 0; i < 30; ++i) {
        next = network_predict(&net, next);
        image newi = save_reconstruction(extractor, &last, next, "newi", i);
        free_image(last);
        last = newi;
    }
}


float_pair get_rnn_vid_datanetwork net, char **files, int n, int batch, int steps) {
    int b;
    assert(net.batch == steps + 1);
    image out_im = get_network_image(&net);
    int output_size = out_im.w * out_im.h * out_im.c;
    printf("%d %d %d\n", out_im.w, out_im.h, out_im.c);
    float *feats = (float *) calloc(net.batch * batch * output_size, sizeof(float));
    for (b = 0; b < batch; ++b) {
        int input_size = net.w * net.h * net.c;
        float *input = (float *) calloc(input_size * net.batch, sizeof(float));
        char *filename = files[rand() % n];
        CvCapture *cap = cvCaptureFromFile(filename);
        int frames = cvGetCaptureProperty(cap, CV_CAP_PROP_FRAME_COUNT);
        int index = rand() % (frames - steps - 2);
        if (frames < (steps + 4)) {
            --b;
            free(input);
            continue;
        }

        printf("frames: %d, index: %d\n", frames, index);
        cvSetCaptureProperty(cap, CV_CAP_PROP_POS_FRAMES, index);

        int i;
        for (i = 0; i < net.batch; ++i) {
            IplImage *src = cvQueryFrame(cap);
            image im = ipl_to_image(src);
            rgbgr_imageim);
            image re = resize_image(im, net.w, net.h);
            //show_image(re, "loaded");
            //cvWaitKey(10);
            memcpy(input + i * input_size, re.data, input_size * sizeof(float));
            free_imageim);
            free_imagere);
        }
        float *output = network_predict(&net, input);

        free(input);

        for (i = 0; i < net.batch; ++i) {
            memcpy(feats + (b + i * batch) * output_size, output + i * output_size, output_size * sizeof(float));
        }

        cvReleaseCapture(&cap);
    }

    //printf("%d %d %d\n", out_im.w, out_im.h, out_im.c);
    float_pair p = {0};
    p.x = feats;
    p.y = feats + output_size * batch; //+ out_im.w*out_im.h*out_im.c;

    return p;
}

image save_reconstruction(network net, image *init, float *feat, char *name, int i) {
    image recon;
    if (init) {
        recon = cv_copy_image(*init);
    } else {
        recon = make_random_image(net.w, net.h, 3);
    }

    image update = make_image(net.w, net.h, 3);
    reconstruct_picture(net, feat, recon, update, .01, .9, .1, 2, 50);
    char buff[1024];
    sprintf(buff, "%s%d", name, i);
    save_image(recon, buff);
    free_image(update);
    return recon;
}

void run_vid_rnn(int argc, char **argv)
{
    if(argc < 4){
        fprintf(stderr, "usage: %s %s [train/test/valid] [cfg] [weights (optional)]\n", argv[0], argv[1]);
        return;
    }

    char *cfg = argv[3];
    char *weights = (argc > 4) ? argv[4] : 0;
    char *filename = (argc > 5) ? argv[5]: 0;
    if(0==strcmp(argv[2], "train")) train_vid_rnn(cfg, weights);
    else if(0==strcmp(argv[2], "generate")) generate_vid_rnn(cfg, weights, filename);
}
#else
void run_vid_rnn(int argc, char **argv){}
#endif

