#######################################################################################################
# Setup clBLAS for LINUX
# wget https://github.com/clMathLibraries/clBLAS/releases/download/v2.12/clBLAS-2.12.0-Linux-x64.tar.gz
# tar zxf clBLAS-2.12.0-Linux-x64.tar.gz
# cd clBLAS-2.12.0-Linux-x64
# cp bin/* /usr/bin/
# cp lib64/* /usr/lib64/
# cp -r lib64/cmake/* /usr/lib64/cmake/
# cp -r lib64/pkgconfig/* /usr/lib64/pkgconfig/
# cp -r include/* /usr/include/
#
# CLBlast can be installed as follows on Ubuntu 20.04:
# sudo add-apt-repository ppa:cnugteren/clblast
# sudo apt-get update
# sudo apt install libclblast-dev
#
# Setup Mali-GPU OpenCL
# git clone https://github.com/krrishnarraj/libopencl-stub
# cd libopencl-stub; mkdir b; cd b; cmake ..; make; make install; cd ..; rm -r b;
#
# Setup VC4CL for RPI
# https://github.com/doe300/VC4CL/wiki/How-to-get
#
# Install OpenCV4 from the Ubuntu 20.04 Repository
# sudo apt update
# sudo apt install libopencv-dev

GPU=1
GPU_FAST=1
GPU_MULTI=0
OPENCV=1
OPENMP=0
# Choose only one (works if GPU=1): NVIDIA or AMD or ARM (for VC4CL or MaliGPU)
NVIDIA=1
AMD=0
ARM=0
BENCHMARK=0
LOSS_ONLY=0
DEBUG=0

VPATH=./src/:./examples
SLIB=libdarknet.so
ALIB=libdarknet.a
EXEC=darknet
OBJDIR=./obj/

CC=gcc
CPP=g++
AR=ar
ARFLAGS=rcs
OPTS=
COMMON= -Iinclude/ -Isrc/
CFLAGS=-Wall -Wno-unknown-pragmas -Wno-unused-variable -Wno-unused-result -Wno-deprecated-declarations -Wno-return-type-c-linkage -Wno-unused-function -Wfatal-errors -fPIC

ifeq ($(ARM), 1)
LDFLAGS= -lm -lpthread
else
LDFLAGS= -lm -lz -lpthread
endif

ifeq ($(DEBUG), 1)
OPTS=-O0 -g
else
ifeq ($(ARM), 1)
OPTS=-O2
else
OPTS=-O3
endif
endif

ifeq ($(OPENMP), 1)
CFLAGS+= -openmp
endif

CFLAGS+=$(OPTS)

ifeq ($(OPENCV), 1) 
COMMON+= -DOPENCV
CFLAGS+= -DOPENCV
LDFLAGS+= `pkg-config --libs opencv4`
COMMON+= `pkg-config --cflags opencv4`
endif

ifeq ($(ARM), 1)
ifeq ($(GPU), 1)
COMMON+= -DGPU -DOPENCL -DCL_TARGET_OPENCL_VERSION=120 -DARM 
CFLAGS+= -DGPU -DOPENCL -DARM -I/usr/include/ -I/usr/local/include/
LDFLAGS+= -L/usr/local/lib -L/usr/lib/arm-linux-gnueabihf -lOpenCL
LDFLAGS+= -L/usr/lib
else
COMMON+= -DCL_TARGET_OPENCL_VERSION=120 -DARM
CFLAGS+= -DARM -I/usr/include/ -I/usr/local/include/
LDFLAGS+= -L/usr/local/lib -L/usr/lib/arm-linux-gnueabihf -lOpenCL
LDFLAGS+= -L/usr/lib
endif
endif

ifeq ($(GPU), 1)
ifeq ($(AMD), 1)
COMMON+= -DGPU -DOPENCL -DCL_TARGET_OPENCL_VERSION=120
CFLAGS+= -DGPU -DOPENCL -I/usr/include/
LDFLAGS+= -L/usr/lib/x86_64-linux-gnu/ -lOpenCL -lclBLAS -L/usr/local/lib
LDFLAGS+= -L/usr/lib64
endif
ifeq ($(NVIDIA), 1)
COMMON+= -DGPU -DOPENCL
CFLAGS+= -DGPU -DOPENCL -I/usr/include/ -I/usr/local/cuda/include/
LDFLAGS+= -L/usr/local/cuda/lib64 -lOpenCL -L/usr/lib64 -lclBLAS -L/usr/local/lib
LDFLAGS+= -L/usr/lib64
endif
endif

ifeq ($(GPU_FAST), 1)
COMMON+= -DGPU_FAST
CFLAGS+= -DGPU_FAST
endif

ifeq ($(GPU_MULTI), 1)
COMMON+= -DGPU_MULTI
CFLAGS+= -DGPU_MULTI
endif

ifeq ($(BENCHMARK), 1)
COMMON+= -DBENCHMARK
CFLAGS+= -DBENCHMARK
endif

ifeq ($(LOSS_ONLY), 1)
COMMON+= -DLOSS_ONLY
CFLAGS+= -DLOSS_ONLY
endif

OBJ=gemm.o utils.o opencl.o deconvolutional_layer.o convolutional_layer.o list.o image.o iseg_layer.o activations.o im2col.o col2im.o blas.o crop_layer.o dropout_layer.o maxpool_layer.o softmax_layer.o data.o matrix.o network.o connected_layer.o cost_layer.o parser.o option_list.o detection_layer.o route_layer.o box.o normalization_layer.o avgpool_layer.o layer.o local_layer.o shortcut_layer.o activation_layer.o rnn_layer.o gru_layer.o crnn_layer.o demo.o batchnorm_layer.o region_layer.o reorg_layer.o tree.o  lstm_layer.o yolo_layer.o yolo4_layer.o upsample_layer.o logistic_layer.o l2norm_layer.o image_opencv.o system.o
EXECOBJA=captcha.o lsd.o cgan.o super.o art.o tag.o cifar.o go.o instance-segmenter.o rnn.o segmenter.o regressor.o classifier.o coco.o yolo.o detector.o nightmare.o attention.o darknet.o
ifeq ($(GPU), 1) 
LDFLAGS+= -lstdc++
OBJ+=convolutional_kernels.o deconvolutional_kernels.o activation_kernels.o im2col_kernels.o col2im_kernels.o blas_kernels.o crop_layer_kernels.o dropout_layer_kernels.o maxpool_layer_kernels.o avgpool_layer_kernels.o
else
#LDFLAGS+= -lstdc++
OBJ+=cpu.o
endif

EXECOBJ = $(addprefix $(OBJDIR), $(EXECOBJA))
OBJS = $(addprefix $(OBJDIR), $(OBJ))
DEPS = $(wildcard src/*.h) Makefile include/darknet.h

#all: obj backup results $(SLIB) $(ALIB) $(EXEC)
all: obj  results $(SLIB) $(ALIB) $(EXEC)


$(EXEC): $(EXECOBJ) $(ALIB)
	$(CC) $(COMMON) $(CFLAGS) $^ -o $@ $(LDFLAGS) $(ALIB)

$(ALIB): $(OBJS)
	$(AR) $(ARFLAGS) $@ $^

$(SLIB): $(OBJS)
	$(CC) $(CFLAGS) -shared $^ -o $@ $(LDFLAGS)

$(OBJDIR)%.o: %.cpp $(DEPS)
	$(CPP) $(COMMON) $(CFLAGS) -c $< -o $@

$(OBJDIR)%.o: %.c $(DEPS)
	$(CC) $(COMMON) $(CFLAGS) -c $< -o $@

obj:
	mkdir -p obj
backup:
	mkdir -p backup
results:
	mkdir -p results

.PHONY: clean

clean:
	rm -rf $(OBJS) $(SLIB) $(ALIB) $(EXEC) $(EXECOBJ)
