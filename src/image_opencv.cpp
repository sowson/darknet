#ifdef OPENCV
#ifdef __cplusplus

#include "stdio.h"
#include "stdlib.h"

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "image.h"

using namespace cv;

extern "C" {

image copy_image_cv(image p)
{
    image copy;
    copy.data = (float *)calloc(p.h*p.w*p.c, sizeof(float));
    memcpy(copy.data, p.data, p.h*p.w*p.c*sizeof(float));
    copy.h = p.h;
    copy.w = p.w;
    copy.c = p.c;
    return copy;
}

IplImage *image_to_ipl_cv(image im)
{
    int x,y,c;
    IplImage *disp = cvCreateImage(cvSize(im.w,im.h), IPL_DEPTH_8U, im.c);
    int step = disp->widthStep;
    for(y = 0; y < im.h; ++y){
        for(x = 0; x < im.w; ++x){
            for(c= 0; c < im.c; ++c){
                float val = im.data[c*im.h*im.w + y*im.w + x];
                disp->imageData[y*step + x*im.c + c] = (unsigned char)(val*255);
            }
        }
    }
    return disp;
}

image make_empty_image_cv(int w, int h, int c)
{
    image out;
    out.data = 0;
    out.h = h;
    out.w = w;
    out.c = c;
    return out;
}

image make_image_cv(int w, int h, int c)
{
    image out = make_empty_image_cv(w,h,c);
    out.data = (float *)calloc(h*w*c, sizeof(float));
    return out;
}

void ipl_into_image_cv(IplImage* src, image im)
{
    unsigned char *data = (unsigned char *)src->imageData;
    int h = src->height;
    int w = src->width;
    int c = src->nChannels;
    int step = src->widthStep;
    int i, j, k;

    for(i = 0; i < h; ++i){
        for(k= 0; k < c; ++k){
            for(j = 0; j < w; ++j){
                im.data[k*w*h + i*w + j] = data[i*step + j*c + k]/255.;
            }
        }
    }
}

image ipl_to_image_cv(IplImage* src)
{
    int h = src->height;
    int w = src->width;
    int c = src->nChannels;
    image out = make_image_cv(w, h, c);
    ipl_into_image_cv(src, out);
    return out;
}

void free_image_cv(image m)
{
    if(m.data){
        free(m.data);
    }
}

void rgbgr_image_cv(image im)
{
    int i;
    for(i = 0; i < im.w*im.h; ++i){
        float swap = im.data[i];
        im.data[i] = im.data[i+im.w*im.h*2];
        im.data[i+im.w*im.h*2] = swap;
    }
}

void constrain_image_cv(image im)
{
    int i;
    for(i = 0; i < im.w*im.h*im.c; ++i){
        if(im.data[i] < 0) im.data[i] = 0;
        if(im.data[i] > 1) im.data[i] = 1;
    }
}

Mat image_to_mat_cv(image img)
{
    int channels = img.c;
    int width = img.w;
    int height = img.h;
    Mat mat = Mat(height, width, CV_8UC(channels));
    int step = mat.step;
    for (int y = 0; y < img.h; ++y) {
        for (int x = 0; x < img.w; ++x) {
            for (int c = 0; c < img.c; ++c) {
                float val = img.data[c*img.h*img.w + y*img.w + x];
                mat.data[y*step + x*img.c + c] = (unsigned char)(val * 255);
            }
        }
    }
    return mat;
}

image mat_to_image_cv(Mat mat)
{
    int w = mat.cols;
    int h = mat.rows;
    int c = mat.channels();
    image im = make_image_cv(w, h, c);
    unsigned char *data = (unsigned char *)mat.data;
    int step = mat.step;
    for (int y = 0; y < h; ++y) {
        for (int k = 0; k < c; ++k) {
            for (int x = 0; x < w; ++x) {
                im.data[k*w*h + y*w + x] = data[y*step + x*c + k] / 255.0f;
            }
        }
    }
    return im;
}

void *open_video_stream(const char *f, int c, int w, int h, int fps)
{
    VideoCapture *cap;
    if(f) cap = new VideoCapture(f);
    else cap = new VideoCapture(c);
    if(!cap->isOpened()) return 0;
    if(w) cap->set(CV_CAP_PROP_FRAME_WIDTH, w);
    if(h) cap->set(CV_CAP_PROP_FRAME_HEIGHT, w);
    if(fps) cap->set(CV_CAP_PROP_FPS, w);
    return (void *) cap;
}

image get_image_from_stream_ocv(void *p)
{
    VideoCapture *cap = (VideoCapture *)p;
    Mat m;
    *cap >> m;
    if(m.empty()) return make_empty_image_cv(0,0,0);
    return mat_to_image_cv(m);
}

image load_image_ocv(char *filename, int channels)
{
    int flag = -1;
    if (channels == 0) flag = -1;
    else if (channels == 1) flag = 0;
    else if (channels == 3) flag = 1;
    else {
        fprintf(stderr, "OpenCV can't force load with %d channels\n", channels);
    }
    Mat m;
    m = imread(filename, flag);
    if(!m.data){
        fprintf(stderr, "Cannot load image \"%s\"\n", filename);
        char buff[256];
        sprintf(buff, "echo %s >> bad.list", filename);
        system(buff);
        return make_image_cv(10,10,3);
        //exit(0);
    }
    image im = mat_to_image_cv(m);
    return im;
}

int show_image_frame_cv(image im, const char* name, int ms)
{
    Mat m = image_to_mat_cv(im);
    imshow(name, m);
    int c = waitKey(ms);
    if (c != -1) c = c%256;
    return c;
}

void make_window_cv(char *name, int w, int h, int fullscreen)
{
    namedWindow(name, WINDOW_NORMAL);
    if (fullscreen) {
        setWindowProperty(name, CV_WND_PROP_FULLSCREEN, CV_WINDOW_FULLSCREEN);
    } else {
        resizeWindow(name, w, h);
        if(strcmp(name, "Demo") == 0) moveWindow(name, 0, 0);
    }
}

void blur_image_and_save_cv(image im, int num, int classes, detection *dets, float thresh, const char* fname)
{
    Mat m = image_to_mat_cv(im);

    int i,j,classi;
    for(i = 0; i < num; ++i) {
        classi = -1;
        for (j = 0; j < classes; ++j) {
            if (dets[i].prob[j] > thresh) {
                if (classi < 0) {
                    classi = j;
                }
            }
        }
        if (classi >= 0) {
            box bx = dets[i].bbox;

            int l = (bx.x - bx.w / 2.) * im.w + 1;
            int r = (bx.x + bx.w / 2.) * im.w - 1;
            int t = (bx.y - bx.h / 2.) * im.h + 1;
            int b = (bx.y + bx.h / 2.) * im.h - 1;

            //printf("%i %i %i %i\n", l, t, r, b);

            Point tl = Point(l, t);
            Point br = Point(r, b);
            Rect roi = Rect(tl, br);

            GaussianBlur(m(roi), m(roi), Size(27, 27), 0, 0);
        }
    }

    im = mat_to_image_cv(m);
    if (im.c == 3) rgbgr_image_cv(im);
    char buff[512];
    sprintf(buff, "%s.jpg", fname);
    imwrite(buff, image_to_mat_cv(im));
    free_image_cv(im);
}

void save_image_jpg_cv(image p, const char *name) {
    /*
    image im = copy_image_cv(p);
    Mat mat = image_to_mat_cv(im);

    char buff[256];
    sprintf(buff, "%s.jpg", name);

    imwrite(buff, mat);
    */
}

}

#endif
#endif
