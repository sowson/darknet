# Darknet v1.1.1 — AI CNN Computer Vision Engine

See also new [iChess.io v12.27](https://github.com/sowson/ichess.io)!!!

📄 This work is described in the scientific paper: https://doi.org/10.1002/cpe.6936

## 📦 Platform Support

- ✅ macOS (Intel / Apple Silicon)
- ✅ Ubuntu Linux 20.04+
- ⚠️ Windows 10/11 (experimental OpenCL build)

### Windows Build Guide:
https://iblog.isowa.io/2021/11/20/darknet-on-opencl-on-windows-11-x64

## 🧠 Training & Optimization Tips

- Run from **RAMDisk** to reduce disk wear and speed up training/inference:
    - Linux: `sudo mount -t tmpfs -o size=4096M tmpfs /your/ramdisk`
    - macOS: `diskutil erasevolume HFS+ "ramdisk" $(hdiutil attach -nomount ram://8388608)`

- Replace clBLAS with **CLBlast** for improved GEMM performance:
```bash
git apply patches/clblast.patch
```

## 🔗 Related Projects

- [Darknet-vNext](https://github.com/sowson/darknet-vNext) — CUDA-enhanced variant
- [YOLO on OpenCL](https://iblog.isowa.io/2020/07/02/the-multi-gpu-set-idea)

## 📽️ Demos & Videos

- [4 x AMD Radeon RX 6900 XT Demo (macOS)](https://www.youtube.com/watch?v=W6VOLjgwKNI)
- [YOLO2 Multi-GPU Result](https://www.youtube.com/watch?v=o-PV3vmfP-0)
- [YOLO Training](https://www.youtube.com/watch?v=Mxw7XkFBFPc)

## 🙏 Acknowledgements

Created by **Piotr Sowa** — AI researcher, GPU software engineer, and creator of [iChess.io](https://iChess.io).
More information and tutorials at [iBlog.isowa.io](https://iblog.isowa.io).

---

For citations, academic usage, or collaboration inquiries, feel free to reach out via GitHub or LinkedIn.
