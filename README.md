# 🚀 CUDA + PyTorch + VS Code GPU Setup (Windows)

> 📌 **Author:** Dishanand Jayeprokash  
> 🗓️ **Created:** 22 July 2025  
> 🔗 **Reference:** [StackOverflow: PyTorch CUDA False](https://stackoverflow.com/questions/60987997/why-torch-cuda-is-available-returns-false-even-after-installing-pytorch-with)

---

<p align="center">
  <img src="images/nvidia_logo.png" alt="NVIDIA Logo" width="120"/>
  &nbsp;&nbsp;&nbsp;
  <img src="images/vscode_logo.png" alt="VS Code Logo" width="100"/>
</p>




---

## 📚 Table of Contents

1. [🖥️ Install NVIDIA GPU](#️-install-your-nvidia-graphics-card-windows-10)
2. [⚙️ Install NVIDIA Drivers](#-install-nvidia-drivers)
3. [🧠 Install CUDA Toolkit](#-install-nvidia-cuda-toolkit)
4. [💻 Enable GPU Acceleration in VS Code](#-enable-gpu-acceleration-in-visual-studio-code)
5. [🔥 Install PyTorch with CUDA Support](#-install-pytorch-with-cuda-support)
6. [🔍 Check PyTorch GPU Access](#-check-gpu-detection-in-pytorch)
7. [🐛 Troubleshooting Common Issues](#-troubleshooting-mismatched-cuda--cpu-only-build)
8. [📘 Detailed Setup Reference](#-detailed-setup-reference)
9. [🎉 Conclusion](#-conclusion)
10. [🧊 Tips](#-tips)
11. [💬 Feedback](#-feedback)


---

## 🖥️ Install Your NVIDIA Graphics Card (Windows 10)

> Plug in your GPU and ensure your power supply unit (PSU) and motherboard support the card.

---

## ⚙️ Install NVIDIA Drivers

🔗 [Download NVIDIA Drivers](https://www.nvidia.com/en-us/drivers/)

✅ **Recommended: Studio Drivers**
- **Stable** for long training jobs
- **Optimized** for machine learning & creative workloads
- **Less frequent updates** → fewer disruptions

---

## 🧠 Install NVIDIA CUDA Toolkit

🔗 [Download CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit)

### ✅ Verify Installation

Open the **VS Code terminal** and run:

```bash
nvcc --version
````

> If CUDA is installed properly, this will return the CUDA version (e.g., 12.9)

### 💡 Why You Need CUDA Toolkit

* Enables frameworks like PyTorch to access GPU acceleration
* Comes with libraries like `cuDNN`, `cuBLAS` for tensor operations
* CUDA compiler `nvcc` enables GPU code compilation

---

## 💻 Enable GPU Acceleration in Visual Studio Code

### 1. Set High Performance Mode

1. Open **Windows Graphics Settings**
2. Click **Browse**, find `Code.exe` (usually in `AppData\Local\Programs\Microsoft VS Code`)
3. Set it to **High Performance**

### 2. Turn on GPU Acceleration in VS Code

1. Go to **File > Preferences > Settings**
2. Search for: `GPU Acceleration`
3. Enable: **Terminal › Integrated › Gpu Acceleration** → Set to `"on"`

---

## 🔥 Install PyTorch with CUDA Support

```bash
pip install torch torchvision torchaudio
```

📦 Alternatively, using `requirements.txt`:

```bash
pip install -r requirements.txt
```
The detailed explanation to build the requirements.txt file is given in this repo:
[requirements.txt](https://github.com/djayepro3/Windows-Venv-Python-Setup/blob/main/README.md#-install-required-packages)

---

## 🔍 Check GPU Detection in PyTorch

Run this Python script to verify:

```python
import torch

print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA version:", torch.version.cuda)
print("Device name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU detected")
```

### ✅ Expected Output

```text
PyTorch version: 2.9.0.dev20250716+cu129
CUDA available: True
CUDA version: 12.9
Device name: NVIDIA GeForce RTX 2070
```

---

## 🐛 Troubleshooting Mismatched CUDA / CPU-only Build

### If your output looks like this:

```text
PyTorch version: 2.7.1+cpu
CUDA available: False
CUDA version: None
Device name: No GPU detected
```

### 🔧 Try These Steps

1. **Check if drivers are installed:**

```bash
nvidia-smi
nvidia-smi -q  # Detailed info
```

2. **Verify cuDNN is enabled:**

```python
import torch
print(torch.backends.cudnn.enabled)
```

3. **Uninstall and reinstall PyTorch:**
   💥 **Nightly Build for CUDA 12.9:**

```bash
pip uninstall torch torchvision torchaudio
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu129
```

4. **Restart VS Code and reactivate virtual environment:**

```bash
deactivate
venv\Scripts\activate
```

5. **Run the GPU check again.**


---

## 📘 Detailed Setup Reference

📄 For a more in-depth explanation, you can view the full setup guide here:  
[**CUDA_PyTorch_VSCode.txt**](setup/CUDA_PyTorch_VSCode.txt)

---

## 🎉 Conclusion

If you see:

```text
PyTorch version: 2.9.0.dev20250716+cu129
CUDA available: True
CUDA version: 12.9
Device name: NVIDIA GeForce RTX 2070
```

✅ You are ready to build and train deep learning models with full GPU acceleration using PyTorch + CUDA on Windows!

---

## 🧊 Tips

* Avoid mixing CUDA versions — keep your driver, toolkit, and PyTorch in sync.
* Nightly builds are for early adopters — upgrade once stable release supports your CUDA version.
* Use Studio Drivers for development; Game Ready Drivers are tuned for performance but less stable.

---

## 💬 Feedback

If you encounter issues or have suggestions, feel free to open an issue or fork the repo and contribute!

To clon the repo, follow these steps:

1. Clone the repository:
    ```bash
    git clone https://github.com/djayepro3/CUDA-PyTorch-VSCode-GPU-Setup
    ```
2. Navigate to the project directory:
    ```bash
    cd CUDA-PyTorch-VSCode-GPU-Setup
    ```

