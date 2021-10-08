# EXPERIMENTAL Build on Windows, I know! But why not once is so hard?!

### Make all of the artefacts by yourself and figured out more!!

0) Clone recursive by command: git clone --recursive https://github.com/sowson/darknet
1) Install MSVC++ from: https://aka.ms/vs/16/release/vc_redist.x64.exe
2) Install Visual Studio 2019 with C/C++ support
3) Install Windows 8.1 SDK from https://developer.microsoft.com/en-us/windows/downloads/sdk-archive
4) Clone / Download and Build from: https://github.com/BrianGladman/pthreads
5) Clone / Download and Build from: https://github.com/robinrowe/libunistd
6) Clone / Download and Use from: https://github.com/nothings/stb
7) Install Python from: https://www.python.org/downloads/windows/
8) Clone / Download and Build clBLAS from: https://github.com/sowson/clBLAS

### Some advices I learn when I was trying to build this Win32

Put into your PATH in OS the all 3rdparty folders with DLLs

- clBLAS
- pthread
- OpenCV

### Please do not blame me... it is still experimental on Win32

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

https://iblog.isowa.io/2020/07/02/the-multi-gpu-set-idea/

[![OpenCL YOLO2 Training Result](https://iblog.isowa.io/wp-content/uploads/2020/06/gitbug-image.jpg)](https://www.youtube.com/watch?v=_dNYNYHXHHo)

https://iblog.isowa.io/2020/06/22/gpu-opencl-fine-tuning-problem-solution/

https://iblog.isowa.io/2020/05/31/ph-d-hanna-hackintosh-is-ready/

[![PhD Progress from May 27th 2020 Update Keynote](https://iblog.isowa.io/wp-content/uploads/2020/05/gitbug-image.jpg)](https://www.youtube.com/watch?v=qfCWYVnJrjQ)

https://iblog.isowa.io/2020/04/29/darknet-in-opencl-on-beagleboard-ai/

[![PhD Progress from March 8th 2020 Update Keynote](https://iblog.isowa.io/wp-content/uploads/2020/03/gitbug-image.jpg)](https://www.youtube.com/watch?v=exuPfFtbwgU)

https://iblog.isowa.io/2020/03/03/is-opencl-beats-cuda/

https://iblog.isowa.io/2020/03/02/hania-pc-well-it-needs-macos/

https://iblog.isowa.io/2020/02/08/pc-for-phd-studies/

https://iblog.isowa.io/2020/01/04/gpu-opencl-fine-tuning-problem/

https://iblog.isowa.io/2019/12/29/darknet-cuda-vs-opencl-and-cpu-vs-nvidia-vs-amd/

https://iblog.isowa.io/2019/11/05/gpu-computing-on-opencl/

https://iblog.isowa.io/2019/08/18/the-fastest-darknet-in-opencl-on-the-planet/

https://iblog.isowa.io/2019/02/02/darknet-in-opencl-on-asus-thinker-board-s/

[![DarkNet Training](https://img.youtube.com/vi/Mxw7XkFBFPc/0.jpg)](https://www.youtube.com/watch?v=Mxw7XkFBFPc)

https://iblog.isowa.io/2018/08/01/darknet-in-opencl/

https://iblog.isowa.io/2018/05/26/darknet-training/ 

Thanks!
