# Build on macOS or Ubuntu 20.04

0) Step by step in command prompt guide: https://iblog.isowa.io/2018/05/26/darknet-training

# Build on Windows 10 or 11 x64

### Make all the artefacts by yourself and figured out more!!

#### First got from the Internet:

0) Clone recursive by command: git clone --recursive https://github.com/sowson/darknet

1) Install MSVC++ from: https://aka.ms/vs/16/release/vc_redist.x64.exe

2) Install Windows 10 or 11 SDK from: https://developer.microsoft.com/en-us/windows/downloads/sdk-archive

3) Install Visual Studio 2019 with C/C++ support (optional but useful)

#### If you want to rebuild things from 3rdparty folder on your own:

4) Clone / Download and Build from: https://github.com/BrianGladman/pthreads

5) Clone / Download and Build from: https://github.com/robinrowe/libunistd

6) Clone / Download and Use from: https://github.com/nothings/stb

#### This is an open issue, I do not know how to build this one:

7) Clone / Download and Build clBLAS from: https://github.com/sowson/clBLAS

#### Needed by some build scripts as an interpreters for scripts:

8) Install Python from: https://www.python.org/downloads/windows

9) Install CMake for Windows from: https://cmake.org/download

#### Build process on Windows 10 can be done in CLion or Visual Studio 2019

#### Please do not blame me... it is still experimental on Windows 10 x64

#### To build as example in the darknet directory in the Git Command Line:

mkdir build

cd build

cmake -S ../ -B ./

cmake --build ./ --config Release --target darknet

cp Release/darknet.exe ../darknet.exe

cp ../3rdparty/clBLAS/clBLAS.dll ..

cp ../3rdparty/pthreads/pthreads.dll ..

cd ..

./darknet.exe # ;-).

# Take a look 4 x GPUs on macOS (click on img to see video)

[![4 x AMD Radeon RX 6900 XT on macOS 11.5.2](https://iblog.isowa.io/wp-content/uploads/2021/08/moria-scaled.jpeg)](https://www.youtube.com/watch?v=W6VOLjgwKNI)

# CLBlast instead of clBLAS for GEMM

git apply patches/clblast.patch

# Darknet-vNext

[DarkNet-vNext Link](https://github.com/sowson/darknet-vNext) If you are looking for engine that has all the same functions, but it is FASTER!

# OpenCV 4

This engine runs on OpenCV v4! But, OpenCV v3 is also fine!

# YOLO4 on OpenCL

YOLO4 elements are supported, remember in CFG file to use [yolo4] instead of [yolo] to make it work!

# YOLO1, YOLO2, YOLO3 on OpenCL

[![OpenCL YOLO2 Training Multi-GPU-SET](https://iblog.isowa.io/wp-content/uploads/2020/07/gitbug-img.jpg)](https://www.youtube.com/watch?v=o-PV3vmfP-0)

https://iblog.isowa.io/2020/07/02/the-multi-gpu-set-idea

[![OpenCL YOLO2 Training Result](https://iblog.isowa.io/wp-content/uploads/2020/06/gitbug-image.jpg)](https://www.youtube.com/watch?v=_dNYNYHXHHo)

https://iblog.isowa.io/2020/06/22/gpu-opencl-fine-tuning-problem-solution

https://iblog.isowa.io/2020/05/31/ph-d-hanna-hackintosh-is-ready

[![PhD Progress from May 27th 2020 Update Keynote](https://iblog.isowa.io/wp-content/uploads/2020/05/gitbug-image.jpg)](https://www.youtube.com/watch?v=qfCWYVnJrjQ)

https://iblog.isowa.io/2020/04/29/darknet-in-opencl-on-beagleboard-ai

[![PhD Progress from March 8th 2020 Update Keynote](https://iblog.isowa.io/wp-content/uploads/2020/03/gitbug-image.jpg)](https://www.youtube.com/watch?v=exuPfFtbwgU)

https://iblog.isowa.io/2020/03/03/is-opencl-beats-cuda

https://iblog.isowa.io/2020/03/02/hania-pc-well-it-needs-macos

https://iblog.isowa.io/2020/02/08/pc-for-phd-studies

https://iblog.isowa.io/2020/01/04/gpu-opencl-fine-tuning-problem

https://iblog.isowa.io/2019/12/29/darknet-cuda-vs-opencl-and-cpu-vs-nvidia-vs-amd

https://iblog.isowa.io/2019/11/05/gpu-computing-on-opencl

https://iblog.isowa.io/2019/08/18/the-fastest-darknet-in-opencl-on-the-planet

https://iblog.isowa.io/2019/02/02/darknet-in-opencl-on-asus-thinker-board-s

[![DarkNet Training](https://img.youtube.com/vi/Mxw7XkFBFPc/0.jpg)](https://www.youtube.com/watch?v=Mxw7XkFBFPc)

https://iblog.isowa.io/2018/08/01/darknet-in-opencl

https://iblog.isowa.io/2018/05/26/darknet-training 

Thanks!
