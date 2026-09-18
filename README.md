# 🚀 CUDA + PyTorch + VS Code GPU Setup (Windows + Linux)

> 📌 **Author:** Dishanand Jayeprokash
> 🗓️ **Created:** 17 July 2025
> ✏️ **Last Modified:** 18 September 2026
> 🔗 **Reference:** [StackOverflow: PyTorch CUDA False](https://stackoverflow.com/questions/60987997/why-torch-cuda-is-available-returns-false-even-after-installing-pytorch-with)

---

<p align="center">
  <img src="images/nvidia_logo.png" alt="NVIDIA Logo" width="120"/>
  &nbsp;&nbsp;&nbsp;
  <img src="images/vscode_logo.png" alt="VS Code Logo" width="100"/>
</p>

---

## 📚 Table of Contents

### 🖥️ Windows

1. [Install NVIDIA GPU](#️-install-your-nvidia-graphics-card)
2. [Install NVIDIA Drivers](#-install-nvidia-drivers)
3. [Verify the NVIDIA Driver](#-verify-the-nvidia-driver)
4. [Install CUDA Toolkit](#-install-nvidia-cuda-toolkit)
5. [Install Visual Studio Code](#-install-visual-studio-code)
6. [Set Up a Python Virtual Environment](#-set-up-a-python-virtual-environment)
7. [Install PyTorch with CUDA Support](#-install-pytorch-with-cuda-support)
8. [Check PyTorch GPU Access](#-check-gpu-detection-in-pytorch)
9. [Run a Real GPU Computation Test](#-run-a-real-gpu-computation-test)

### 🐧 Linux

10. [Linux Overview](#-linux-overview)
11. [Check Your Linux System](#-check-your-linux-system)
12. [Choose the Correct Linux Path](#-choose-the-correct-linux-path)
13. [Path A — NVIDIA Driver Already Installed](#-path-a--nvidia-driver-already-installed)
14. [Path B — NVIDIA Driver Not Installed](#-path-b--nvidia-driver-not-installed)
15. [Ubuntu Driver Installation](#-ubuntu-nvidia-driver-installation)
16. [Debian Driver Installation](#-debian-nvidia-driver-installation)
17. [Fedora Driver Installation](#-fedora-nvidia-driver-installation)
18. [Arch Linux Driver Installation](#-arch-linux-nvidia-driver-installation)
19. [Verify the Linux NVIDIA Driver](#-verify-the-linux-nvidia-driver)
20. [Understand NVIDIA Driver vs CUDA Toolkit](#-understand-nvidia-driver-vs-cuda-toolkit)
21. [Install CUDA Toolkit on Linux](#-install-cuda-toolkit-on-linux)
22. [Install Visual Studio Code on Linux](#-install-visual-studio-code-on-linux)
23. [Set Up a Python Virtual Environment on Linux](#-set-up-a-python-virtual-environment-on-linux)
24. [Install PyTorch with CUDA on Linux](#-install-pytorch-with-cuda-on-linux)

### 🔍 Verification and Troubleshooting

25. [GPU Verification Checklist](#-gpu-verification-checklist)
26. [Full Troubleshooting Tree](#-full-troubleshooting-tree)
27. [NVIDIA Driver Troubleshooting](#-nvidia-driver-troubleshooting)
28. [PyTorch CUDA Troubleshooting](#-pytorch-cuda-troubleshooting)
29. [Secure Boot Troubleshooting](#-secure-boot-troubleshooting)
30. [Kernel / DKMS Troubleshooting](#-kernel--dkms-troubleshooting)
31. [GPU Architecture Compatibility](#-gpu-architecture-compatibility)
32. [Avoid Mixing Installation Methods](#-avoid-mixing-installation-methods)

### 📘 Reference

33. [Detailed Setup Reference](#-detailed-setup-reference)
34. [Useful Commands](#-useful-commands)
35. [Tips](#-tips)
36. [Sources and Official Documentation](#-sources-and-official-documentation)
37. [Conclusion](#-conclusion)
38. [Feedback](#-feedback)
39. [Clone This Repository](#-clone-this-repository)

---

# 🖥️ WINDOWS

## 🖥️ Install Your NVIDIA Graphics Card

Plug in your NVIDIA GPU and make sure:

* The motherboard supports the GPU.
* The power supply unit (PSU) provides sufficient power.
* The GPU is correctly seated and connected.
* The BIOS/UEFI detects the GPU.

You can verify the GPU from Windows Device Manager:

```text
Device Manager
└── Display adapters
    └── NVIDIA GPU
```

---

## ⚙️ Install NVIDIA Drivers

🔗 [Download NVIDIA Drivers](https://www.nvidia.com/en-us/drivers/)

NVIDIA provides different driver branches for different use cases.

### Game Ready Driver

Designed primarily for gaming and newly released games.

### Studio Driver

Designed for professional creative applications and workflows.

> 💡 **Note:** Both driver branches provide NVIDIA GPU support. Choose the driver branch appropriate for your workload rather than assuming that one is universally better for CUDA.

---

## 🔍 Verify the NVIDIA Driver

Open the **VS Code terminal**, PowerShell, Command Prompt, or another terminal and run:

```powershell
nvidia-smi
```

A successful output should contain information similar to:

```text
+-----------------------------------------------------------------------------+
| NVIDIA-SMI ...                  Driver Version: ...    CUDA Version: ...    |
|-------------------------------+----------------------+----------------------|
| GPU Name                      | Bus-Id               | Memory-Usage         |
+-------------------------------+----------------------+----------------------+
```

### What `nvidia-smi` tells you

`nvidia-smi` is primarily a **driver/GPU verification tool**.

It can show:

* GPU model
* NVIDIA driver version
* GPU memory usage
* GPU utilization
* Running GPU processes
* The maximum CUDA version supported by the installed driver

> ⚠️ **Important:** The CUDA version shown by `nvidia-smi` should **not** be interpreted as proof that the corresponding CUDA Toolkit is installed.

---

# 🧠 Install NVIDIA CUDA Toolkit

🔗 [NVIDIA CUDA Toolkit](https://developer.nvidia.com/cuda-downloads)

The CUDA Toolkit is mainly required when you need CUDA development tools such as:

* `nvcc`
* CUDA headers
* CUDA development libraries
* CUDA debugging and profiling tools
* Compiling CUDA applications or extensions

### Verify the CUDA Toolkit

```powershell
nvcc --version
```

or:

```powershell
nvcc -V
```

Example:

```text
Cuda compilation tools, release 13.x, V13.x.x
```

### Driver vs Toolkit

Think of the stack as:

```text
NVIDIA GPU
    │
    ▼
NVIDIA Driver
    │
    ├── Allows the operating system to communicate with the GPU
    │
    ▼
CUDA Runtime
    │
    ▼
PyTorch CUDA build

CUDA Toolkit
    │
    ├── nvcc
    ├── CUDA headers
    ├── CUDA development libraries
    └── CUDA development tools
```

> 💡 **Important:** A normal user running a pre-built PyTorch CUDA package does not necessarily need the full CUDA Toolkit installed on the operating system.

---

# 💻 Install Visual Studio Code

🔗 [Download Visual Studio Code](https://code.visualstudio.com/)

Recommended extensions:

* **Python**
* **Pylance**
* **Jupyter**
* **GitHub Pull Requests and Issues**

---

## 🖥️ Optional: Windows Graphics Settings

For systems with integrated + discrete GPUs, Windows can assign applications to different graphics processors.

1. Open **Settings**
2. Go to **System → Display → Graphics**
3. Add `Code.exe`
4. Select **High performance**

Typical VS Code path:

```text
C:\Users\<username>\AppData\Local\Programs\Microsoft VS Code\Code.exe
```

> 💡 This is a Windows graphics-selection feature. It is separate from PyTorch's CUDA compute support.

---

## ⚡ Optional: VS Code Terminal GPU Acceleration

VS Code supports GPU-accelerated terminal rendering.

Open:

```text
File
→ Preferences
→ Settings
```

Search for:

```text
GPU Acceleration
```

The terminal setting is:

```json
"terminal.integrated.gpuAcceleration": "auto"
```

You can set it explicitly:

```json
"terminal.integrated.gpuAcceleration": "on"
```

> ⚠️ **Important:** VS Code terminal GPU rendering is not what makes PyTorch use CUDA. PyTorch GPU computation is verified through the NVIDIA driver and PyTorch itself.

---

# 🐍 Set Up a Python Virtual Environment

Create a virtual environment:

```powershell
python -m venv .venv
```

Activate it:

```powershell
.venv\Scripts\activate
```

Upgrade `pip`:

```powershell
python -m pip install --upgrade pip
```

Verify:

```powershell
python --version
python -m pip --version
```

---

# 🔥 Install PyTorch with CUDA Support

🔗 [PyTorch Start Locally](https://pytorch.org/get-started/locally/)

Do not assume that:

```powershell
pip install torch torchvision torchaudio
```

always installs the GPU configuration you want.

Use the official PyTorch installation selector and choose:

```text
OS: Windows
Package: Pip
Language: Python
Compute Platform: CUDA
```

### Example: Explicit CUDA 13.2 installation

The official PyTorch wheel repository currently provides CUDA 13.2 builds:

```powershell
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu132
```

### Older GPU / older driver compatibility

For older supported GPUs, CUDA 12.6 remains available:

```powershell
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
```

> 💡 Always verify your GPU architecture and driver requirements before selecting a CUDA build.

---

# 🔍 Check GPU Detection in PyTorch

Run:

```python
import torch

print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("PyTorch CUDA version:", torch.version.cuda)
print("GPU count:", torch.cuda.device_count())

if torch.cuda.is_available():
    print("Device name:", torch.cuda.get_device_name(0))
else:
    print("Device name: No GPU detected")
```

Expected GPU-enabled output:

```text
PyTorch version: 2.14.0+cu132
CUDA available: True
PyTorch CUDA version: 13.2
GPU count: 1
Device name: NVIDIA GeForce RTX ...
```

> 📌 The exact version, GPU name, and CUDA build may differ depending on your installation.

---

# 🧪 Run a Real GPU Computation Test

Detecting the GPU is useful, but performing an operation on the GPU provides a stronger verification.

```python
import torch

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available.")

device = torch.device("cuda")

x = torch.randn(2048, 2048, device=device)
y = torch.randn(2048, 2048, device=device)

z = x @ y

print("Computation successful")
print("Device:", z.device)
print("Tensor shape:", z.shape)
```

Expected:

```text
Computation successful
Device: cuda:0
Tensor shape: torch.Size([2048, 2048])
```

---

# 🐧 LINUX

## 🐧 Linux Overview

Linux NVIDIA setups are different from Windows because the NVIDIA driver is integrated with the Linux kernel, package manager, boot process, Secure Boot configuration, and graphics stack.

Most importantly:

```text
Linux NVIDIA setup
│
├── NVIDIA Driver
│
├── CUDA Toolkit
│
├── Python environment
│
├── PyTorch CUDA build
│
└── VS Code
```

These components should be treated as separate layers.

---

# 🔎 Check Your Linux System

Before installing anything, identify:

### Linux distribution

```bash
cat /etc/os-release
```

or:

```bash
hostnamectl
```

### Kernel

```bash
uname -r
```

### CPU architecture

```bash
uname -m
```

Typical output:

```text
x86_64
```

### PCI devices

```bash
lspci | grep -Ei 'vga|3d|nvidia'
```

Example:

```text
01:00.0 3D controller: NVIDIA Corporation ...
```

### Secure Boot status

```bash
mokutil --sb-state
```

If `mokutil` is unavailable:

```bash
sudo apt install mokutil
```

for Debian/Ubuntu-based systems.

---

# 🧭 Choose the Correct Linux Path

Use this decision tree before installing NVIDIA software:

```text
                         ┌───────────────────────────────┐
                         │ Do you have an NVIDIA GPU?    │
                         └───────────────┬───────────────┘
                                         │
                              ┌──────────┴──────────┐
                              │                     │
                             NO                    YES
                              │                     │
                              ▼                     ▼
                         CUDA/NVIDIA         Is the NVIDIA driver
                         GPU setup not       already working?
                         applicable                │
                                             ┌─────┴─────┐
                                             │           │
                                            YES          NO
                                             │           │
                                             ▼           ▼
                                      PATH A           PATH B
                                  Driver already    Driver must be
                                      works           installed
```

---

# ✅ PATH A — NVIDIA Driver Already Installed

This is the recommended path for systems such as:

* **Pop!_OS NVIDIA**
* A workstation where the NVIDIA driver is already installed
* A previously configured Ubuntu/Debian/Fedora/Arch system
* A machine where `nvidia-smi` already works

## Example: Pop!_OS

Pop!_OS provides a dedicated NVIDIA image for compatible NVIDIA hardware.

🔗 [Pop!_OS Downloads](https://system76.com/pop/)

If you installed an NVIDIA-specific Pop!_OS image and:

```bash
nvidia-smi
```

works correctly, **do not install another NVIDIA driver on top of it**.

Continue with:

```text
nvidia-smi
   │
   ▼
Driver working
   │
   ▼
Create Python environment
   │
   ▼
Install PyTorch CUDA build
   │
   ▼
Run PyTorch verification
```

### Quick verification

```bash
nvidia-smi
```

Then:

```bash
lsmod | grep nvidia
```

Then:

```bash
cat /proc/driver/nvidia/version
```

If these checks are successful, move to the Python/PyTorch section.

---

# ❌ PATH B — NVIDIA Driver Not Installed

If:

```bash
nvidia-smi
```

returns:

```text
command not found
```

or:

```text
NVIDIA-SMI has failed because it couldn't communicate with the NVIDIA driver
```

then the NVIDIA driver needs to be investigated.

> ⚠️ Do **not** immediately run the NVIDIA `.run` installer.
>
> First determine your Linux distribution and use its recommended package-management method.

General Linux flow:

```text
Linux distribution
       │
       ▼
Identify recommended NVIDIA driver
       │
       ▼
Install distro-supported driver
       │
       ▼
Reboot
       │
       ▼
nvidia-smi
       │
       ├── Works → continue
       │
       └── Fails → troubleshoot driver
```

---

# 🟠 Ubuntu NVIDIA Driver Installation

Ubuntu provides the `ubuntu-drivers` tool and the **Additional Drivers** application.

🔗 [Ubuntu NVIDIA Driver Documentation](https://ubuntu.com/desktop/docs/en/latest/how-to/graphics/install-nvidia-drivers/)

### Update the system

```bash
sudo apt update
sudo apt upgrade
```

### Detect available drivers

```bash
ubuntu-drivers devices
```

or:

```bash
ubuntu-drivers list
```

### Install the recommended driver

```bash
sudo ubuntu-drivers install
```

Then reboot:

```bash
sudo reboot
```

Verify:

```bash
nvidia-smi
```

### Why use `ubuntu-drivers`?

Ubuntu recommends its packaged drivers because they are integrated with the Ubuntu release and its package-management system.

> ⚠️ Avoid installing an unrelated NVIDIA driver manually over the Ubuntu package-managed driver unless you have a specific reason and understand the consequences.

---

# 🟣 Debian NVIDIA Driver Installation

Debian uses its own package-management model for NVIDIA.

🔗 [Debian NVIDIA Graphics Drivers](https://wiki.debian.org/NvidiaGraphicsDrivers)

Before installation, check that the required Debian repository components are enabled for your Debian release.

For modern Debian installations, this commonly involves:

```text
main
contrib
non-free
non-free-firmware
```

Check:

```bash
cat /etc/apt/sources.list
```

and/or:

```bash
grep -R "^deb " /etc/apt/sources.list /etc/apt/sources.list.d/
```

Then:

```bash
sudo apt update
```

Debian currently provides an `nvidia-driver` package.

Example:

```bash
sudo apt install nvidia-driver
```

After installation:

```bash
sudo reboot
```

Verify:

```bash
nvidia-smi
```

> 📌 The exact recommended package/module configuration can vary by Debian release, GPU generation, kernel, and Secure Boot configuration. Check Debian's current NVIDIA documentation before making additional changes.

---

# 🔵 Fedora NVIDIA Driver Installation

Fedora uses a different packaging ecosystem from Debian/Ubuntu.

A common community-supported route is **RPM Fusion**.

🔗 [RPM Fusion](https://rpmfusion.org/)

Before installing NVIDIA packages:

```bash
sudo dnf update
```

Follow the current RPM Fusion NVIDIA instructions for your Fedora release.

Common package names include:

```text
akmod-nvidia
nvidia-driver
nvidia-open
```

The exact package choice depends on:

* Fedora release
* GPU generation
* kernel
* NVIDIA driver branch
* open vs proprietary kernel module requirements

After installation:

```bash
sudo reboot
```

Then verify:

```bash
nvidia-smi
```

> ⚠️ Fedora users should follow the current RPM Fusion instructions rather than copying an NVIDIA package command intended for Ubuntu or Debian.

---

# 🟢 Arch Linux NVIDIA Driver Installation

Arch Linux uses its own package model.

🔗 [ArchWiki — NVIDIA](https://wiki.archlinux.org/title/NVIDIA)

For current supported GPUs, Arch provides NVIDIA open-kernel-module packages such as:

```text
nvidia-open
nvidia-open-lts
nvidia-open-dkms
```

The correct package depends on the GPU and kernel.

For example:

```text
Current Linux kernel
└── nvidia-open

Linux LTS kernel
└── nvidia-open-lts

Custom / multiple kernels
└── nvidia-open-dkms
```

Older GPUs may require different or legacy driver branches.

After installation:

```bash
sudo reboot
```

Verify:

```bash
nvidia-smi
```

> 💡 NVIDIA's open kernel modules are not the same thing as the Nouveau driver.

---

# 🔍 Verify the Linux NVIDIA Driver

After installation or reboot:

```bash
nvidia-smi
```

You can also check:

```bash
lsmod | grep nvidia
```

and:

```bash
cat /proc/driver/nvidia/version
```

Check the GPU detected by Linux:

```bash
lspci | grep -Ei 'vga|3d|nvidia'
```

Check kernel:

```bash
uname -r
```

Check Secure Boot:

```bash
mokutil --sb-state
```

### Recommended verification tree

```text
lspci
  │
  ├── NVIDIA GPU detected
  │
  ▼
nvidia-smi
  │
  ├── Works
  │    │
  │    ▼
  │   Driver operational
  │
  └── Fails
       │
       ├── Check lsmod
       ├── Check kernel
       ├── Check Secure Boot
       ├── Check DKMS
       ├── Check driver version
       └── Check Nouveau conflict
```

---

# 🧠 Understand NVIDIA Driver vs CUDA Toolkit

This distinction is fundamental.

## NVIDIA Driver

The NVIDIA driver enables Linux to communicate with the NVIDIA GPU.

Primary verification:

```bash
nvidia-smi
```

## CUDA Toolkit

The CUDA Toolkit provides development tools such as:

```bash
nvcc
```

Primary verification:

```bash
nvcc --version
```

## PyTorch CUDA build

PyTorch can be installed as a pre-built binary containing the CUDA runtime components appropriate for that build.

Primary verification:

```python
torch.cuda.is_available()
```

### The complete model

```text
                       NVIDIA GPU
                           │
                           ▼
                    NVIDIA DRIVER
                           │
                  ┌────────┴────────┐
                  │                 │
                  ▼                 ▼
             CUDA Runtime      CUDA TOOLKIT
                  │                 │
                  │                 ├── nvcc
                  │                 ├── headers
                  │                 └── development tools
                  │
                  ▼
              PYTORCH CUDA
                  │
                  ▼
             GPU COMPUTATION
```

---

# 🧠 Important: `nvidia-smi` CUDA Version vs `nvcc` CUDA Version

These two commands answer different questions.

### `nvidia-smi`

```bash
nvidia-smi
```

Primarily answers:

> What NVIDIA driver is installed, and what CUDA level does that driver support?

### `nvcc`

```bash
nvcc --version
```

Answers:

> What CUDA Toolkit compiler is installed?

Therefore this is possible:

```text
nvidia-smi
    ✅ works

nvcc --version
    ❌ command not found
```

This can be completely normal if the NVIDIA driver is installed but the CUDA Toolkit is not.

---

# 🧰 Install CUDA Toolkit on Linux

🔗 [NVIDIA CUDA Installation Guide for Linux](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/)

NVIDIA supports several installation methods.

The preferred general strategy is:

```text
Distribution package manager
        │
        ├── Debian / Ubuntu
        ├── Fedora / RHEL
        └── SUSE / other supported distros
```

rather than blindly using the standalone `.run` installer.

## Ubuntu / Debian

Follow the NVIDIA CUDA repository instructions for your exact release.

Typical package installation after configuring the NVIDIA CUDA repository:

```bash
sudo apt update
sudo apt install cuda-toolkit
```

Then verify:

```bash
nvcc --version
```

## Fedora / RPM-based distributions

After configuring the appropriate NVIDIA CUDA repository:

```bash
sudo dnf install cuda-toolkit
```

Then:

```bash
nvcc --version
```

### PATH

Depending on the installation method and release, you may need to add the CUDA Toolkit binaries to your PATH.

For example:

```bash
export PATH=/usr/local/cuda/bin:$PATH
```

To make the change persistent:

```bash
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
source ~/.bashrc
```

For Zsh:

```bash
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.zshrc
source ~/.zshrc
```

> ⚠️ Do not blindly add paths until you know where CUDA was installed.

Find CUDA installations:

```bash
ls -ld /usr/local/cuda*
```

---

# 💻 Install Visual Studio Code on Linux

🔗 [VS Code — Linux Installation](https://code.visualstudio.com/docs/setup/linux)

VS Code provides installation methods for several Linux families.

### Ubuntu / Debian

Use the official `.deb` package or Microsoft's repository instructions.

### Fedora / RHEL

Use the official RPM package/repository instructions.

### Arch Linux

Use the appropriate Arch package or community-supported installation method.

After installation:

```bash
code --version
```

---

# ⚡ VS Code GPU Acceleration on Linux

VS Code terminal GPU acceleration is separate from PyTorch CUDA.

The setting is:

```json
"terminal.integrated.gpuAcceleration": "auto"
```

Possible values include:

```json
"terminal.integrated.gpuAcceleration": "auto"
```

or:

```json
"terminal.integrated.gpuAcceleration": "on"
```

or:

```json
"terminal.integrated.gpuAcceleration": "off"
```

> ✅ PyTorch does not require VS Code's terminal renderer to be GPU accelerated.

---

# 🐍 Set Up a Python Virtual Environment on Linux

Create:

```bash
python3 -m venv .venv
```

Activate:

```bash
source .venv/bin/activate
```

Upgrade `pip`:

```bash
python -m pip install --upgrade pip
```

Verify:

```bash
python --version
python -m pip --version
```

Your prompt may now look like:

```text
(.venv) user@computer:~/project$
```

---

# 🔥 Install PyTorch with CUDA on Linux

🔗 [PyTorch — Start Locally](https://pytorch.org/get-started/locally/)

For a CUDA-enabled NVIDIA system:

```bash
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu132
```

### CUDA 12.6 compatibility path

For older GPU architectures or older supported drivers:

```bash
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
```

> 💡 Choose the PyTorch CUDA build based on the combination of:
>
> ```text
> GPU architecture
> +
> NVIDIA driver
> +
> PyTorch support
> ```

Do not choose a CUDA build solely because its number looks similar to `nvidia-smi`.

---

# 🧩 CUDA Toolkit Required or Not?

Use this decision tree:

```text
Do you only want to run PyTorch?
          │
       ┌──┴──┐
      YES    NO
       │      │
       ▼      ▼
Install a   Do you need to
PyTorch     compile CUDA code,
CUDA build  extensions, or
            develop CUDA apps?
                 │
              ┌──┴──┐
             YES    NO
              │      │
              ▼      ▼
       Install CUDA   Toolkit
       Toolkit        optional
```

### Typical ML user

```text
NVIDIA Driver ✅
PyTorch CUDA ✅
CUDA Toolkit ⭕ Optional
```

### CUDA developer

```text
NVIDIA Driver ✅
PyTorch CUDA ✅
CUDA Toolkit ✅
nvcc ✅
```

---

# 🔍 GPU VERIFICATION CHECKLIST

Use the following sequence rather than running random commands.

## Step 1 — Hardware

```bash
lspci | grep -Ei 'vga|3d|nvidia'
```

Expected:

```text
NVIDIA Corporation ...
```

## Step 2 — Driver

```bash
nvidia-smi
```

Expected:

```text
GPU Name
Driver Version
CUDA Version
```

## Step 3 — Kernel module

```bash
lsmod | grep nvidia
```

Expected one or more NVIDIA kernel modules.

## Step 4 — CUDA Toolkit

Only if installed:

```bash
nvcc --version
```

## Step 5 — Python

```bash
python --version
python -m pip --version
```

## Step 6 — PyTorch

```python
import torch

print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA version:", torch.version.cuda)
print("GPU count:", torch.cuda.device_count())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
```

## Step 7 — Actual computation

```python
import torch

assert torch.cuda.is_available(), "CUDA is not available"

x = torch.randn(2048, 2048, device="cuda")
y = torch.randn(2048, 2048, device="cuda")
z = x @ y

print("GPU computation successful")
print("Device:", z.device)
```

---

# 🐛 FULL TROUBLESHOOTING TREE

Use this tree to identify which layer is failing.

```text
┌───────────────────────────────┐
│       NVIDIA GPU present?     │
└───────────────┬───────────────┘
                │
          ┌─────┴─────┐
          │           │
         NO          YES
          │           │
          ▼           ▼
     Check BIOS    `nvidia-smi`
     Hardware      works?
     PCIe             │
                 ┌────┴────┐
                 │         │
                NO        YES
                 │         │
                 ▼         ▼
          DRIVER LAYER   Driver OK
                 │         │
        ┌────────┼───────┐ │
        │        │       │ │
     Secure   Kernel   Nouveau│
      Boot     /DKMS   conflict
        │        │       │ │
        └────────┴───────┘ │
                            ▼
                     PyTorch installed?
                            │
                       ┌────┴────┐
                       │         │
                      NO        YES
                       │         │
                       ▼         ▼
                 Install CUDA   `torch.cuda.is_available()`
                 PyTorch build       │
                                  ┌───┴───┐
                                  │       │
                                 NO      YES
                                  │       │
                                  ▼       ▼
                          PYTORCH LAYER  GPU ready
                                  │
                        ┌─────────┼──────────┐
                        │         │          │
                    CPU-only   Driver     GPU architecture
                      build    mismatch    unsupported
```

---

# 🛠️ NVIDIA DRIVER TROUBLESHOOTING

## Problem

```bash
nvidia-smi
```

fails.

### Check GPU

```bash
lspci | grep -Ei 'vga|3d|nvidia'
```

### Check kernel module

```bash
lsmod | grep nvidia
```

### Check driver module version

```bash
cat /proc/driver/nvidia/version
```

### Check kernel

```bash
uname -r
```

### Check Secure Boot

```bash
mokutil --sb-state
```

### Check DKMS

```bash
dkms status
```

### Check kernel messages

```bash
sudo dmesg | grep -i nvidia
```

---

# 🔐 Secure Boot Troubleshooting

Secure Boot can affect whether a Linux kernel module is allowed to load.

Check:

```bash
mokutil --sb-state
```

Possible output:

```text
SecureBoot enabled
```

or:

```text
SecureBoot disabled
```

If Secure Boot is enabled and the NVIDIA module fails to load:

```text
Secure Boot
     │
     ▼
Kernel module signature
     │
     ├── Valid / trusted
     │      │
     │      ▼
     │   Module loads
     │
     └── Not trusted
            │
            ▼
      NVIDIA module fails
      to load
```

Use the current documentation for your distribution to configure signed modules, MOK enrollment, or another supported solution.

Do not randomly disable Secure Boot simply because `nvidia-smi` fails.

---

# 🔧 KERNEL / DKMS TROUBLESHOOTING

The NVIDIA kernel module must be compatible with the running Linux kernel.

Check:

```bash
uname -r
```

Then:

```bash
dkms status
```

Typical problems include:

```text
Driver installed
      │
      ▼
Kernel updated
      │
      ▼
NVIDIA module not rebuilt
      │
      ▼
Driver unavailable
```

Useful checks:

```bash
lsmod | grep nvidia
```

```bash
modinfo nvidia | head
```

```bash
dkms status
```

On Ubuntu, also check available NVIDIA kernel-module packages:

```bash
apt list --installed | grep linux-modules-nvidia
```

---

# 🐧 Nouveau Driver

Linux systems may contain the open-source **Nouveau** NVIDIA driver.

Nouveau and NVIDIA's proprietary/open NVIDIA kernel modules are different components.

Conceptually:

```text
NVIDIA GPU
   │
   ├── Nouveau
   │
   ├── NVIDIA proprietary kernel module
   │
   └── NVIDIA open kernel module
```

A conflicting graphics configuration can prevent the intended NVIDIA driver from working correctly.

Check loaded graphics modules:

```bash
lsmod | grep -E 'nvidia|nouveau'
```

> ⚠️ Do not disable Nouveau manually unless the current documentation for your distribution and driver requires it.

---

# 🧪 PYTORCH CUDA TROUBLESHOOTING

## Problem

```text
PyTorch version: 2.x.x+cpu
CUDA available: False
CUDA version: None
Device name: No GPU detected
```

The most likely problem is that a **CPU-only PyTorch build** was installed.

Check:

```python
import torch

print(torch.__version__)
print(torch.version.cuda)
print(torch.cuda.is_available())
```

If the version contains:

```text
+cpu
```

and:

```text
torch.version.cuda
```

is:

```text
None
```

you have a CPU build.

---

## Reinstall the CUDA-enabled PyTorch Build

First activate your virtual environment.

### Windows

```powershell
.venv\Scripts\activate
```

### Linux

```bash
source .venv/bin/activate
```

Then remove the existing packages:

```bash
python -m pip uninstall torch torchvision torchaudio
```

Install the desired CUDA build.

### CUDA 13.2

```bash
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu132
```

### CUDA 12.6

```bash
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
```

Then verify again.

---

# 🧭 PyTorch Troubleshooting Decision Tree

```text
torch.cuda.is_available()
          │
      ┌───┴───┐
      │       │
    TRUE     FALSE
      │       │
      ▼       ▼
   GPU OK   Check torch.__version__
              │
              ├── `+cpu`
              │      │
              │      ▼
              │   Install CUDA build
              │
              └── `+cu...`
                     │
                     ▼
                Check `nvidia-smi`
                     │
                ┌────┴────┐
                │         │
               NO        YES
                │         │
                ▼         ▼
          Driver issue  Check compatibility
                         │
                         ├── Driver too old
                         ├── GPU architecture
                         └── Environment issue
```

---

# 🧬 GPU ARCHITECTURE COMPATIBILITY

A newer CUDA/PyTorch build may no longer support every historical NVIDIA GPU architecture.

For example, current PyTorch CUDA binaries have moved toward newer architectures, while CUDA 12.6 remains an important compatibility option for older NVIDIA GPUs.

Conceptually:

```text
Older GPU
   │
   ├── Check architecture
   │
   ├── Check PyTorch support
   │
   ├── Check CUDA build
   │
   └── Check driver
```

Examples of older generations that may require special attention:

```text
Maxwell
Pascal
Volta
```

Current CUDA 13.x PyTorch binaries target newer architectures, while CUDA 12.6 remains available for older supported architectures.

> ⚠️ Do not choose a CUDA version based only on your NVIDIA driver's displayed CUDA number. Check the PyTorch release notes and supported architectures for the exact GPU.

---

# 📦 AVOID MIXING INSTALLATION METHODS

One of the most common sources of Linux GPU problems is mixing incompatible installation methods.

Avoid configurations such as:

```text
Distribution NVIDIA Driver
        +
NVIDIA .run Driver
        +
Multiple CUDA repositories
        +
Random third-party packages
```

Prefer:

```text
Distribution-supported NVIDIA Driver
              +
NVIDIA CUDA Toolkit repository
              +
PyTorch official Python package
```

or:

```text
Existing working NVIDIA Driver
              +
PyTorch official Python package
```

### Recommended principle

```text
Use your Linux distribution's supported driver method
              ↓
Verify `nvidia-smi`
              ↓
Install CUDA Toolkit only when needed
              ↓
Install PyTorch's appropriate CUDA build
```

> ⚠️ NVIDIA's standalone `.run` installer is available for Linux, but it should not be the default beginner installation method when your distribution provides an integrated package-management solution.

---

# 📘 DETAILED SETUP REFERENCE

## NVIDIA

🔗 [NVIDIA Drivers](https://www.nvidia.com/en-us/drivers/)

🔗 [NVIDIA CUDA Toolkit](https://developer.nvidia.com/cuda-downloads)

🔗 [CUDA Installation Guide for Linux](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/)

🔗 [CUDA Compatibility](https://docs.nvidia.com/deploy/cuda-compatibility/)

---

## PyTorch

🔗 [PyTorch Start Locally](https://pytorch.org/get-started/locally/)

🔗 [PyTorch Previous Versions](https://pytorch.org/get-started/previous-versions/)

🔗 [PyTorch Releases](https://github.com/pytorch/pytorch/releases)

---

## Linux Distribution Documentation

### Pop!_OS

🔗 [System76 — Pop!_OS](https://system76.com/pop/)

### Ubuntu

🔗 [Ubuntu NVIDIA Drivers](https://ubuntu.com/desktop/docs/en/latest/how-to/graphics/install-nvidia-drivers/)

### Debian

🔗 [Debian NVIDIA Graphics Drivers](https://wiki.debian.org/NvidiaGraphicsDrivers)

### Fedora

🔗 [RPM Fusion](https://rpmfusion.org/)

### Arch Linux

🔗 [ArchWiki — NVIDIA](https://wiki.archlinux.org/title/NVIDIA)

---

## Visual Studio Code

🔗 [VS Code](https://code.visualstudio.com/)

🔗 [VS Code — Linux Installation](https://code.visualstudio.com/docs/setup/linux)

🔗 [VS Code Terminal Appearance](https://code.visualstudio.com/docs/terminal/appearance)

---

# 🧰 USEFUL COMMANDS

## Hardware

### Linux

```bash
lspci | grep -Ei 'vga|3d|nvidia'
```

### Windows

```powershell
wmic path win32_VideoController get name
```

---

## NVIDIA Driver

```bash
nvidia-smi
```

```bash
nvidia-smi -q
```

```bash
lsmod | grep nvidia
```

```bash
cat /proc/driver/nvidia/version
```

---

## CUDA Toolkit

```bash
nvcc --version
```

```bash
which nvcc
```

```bash
ls -ld /usr/local/cuda*
```

---

## Linux System

```bash
cat /etc/os-release
```

```bash
uname -r
```

```bash
uname -m
```

```bash
mokutil --sb-state
```

```bash
dkms status
```

---

## Python

```bash
python --version
```

```bash
python -m pip --version
```

```bash
python -m pip list
```

---

## PyTorch

```python
import torch

print("PyTorch:", torch.__version__)
print("CUDA:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
print("GPU count:", torch.cuda.device_count())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
```

---

# 🧊 TIPS

### 1. Do not confuse the driver with the Toolkit

```text
nvidia-smi
    ↓
NVIDIA Driver

nvcc --version
    ↓
CUDA Toolkit

torch.cuda.is_available()
    ↓
PyTorch CUDA accessibility
```

---

### 2. Do not assume CUDA versions must be identical

For example:

```text
Driver CUDA capability
        ≠
CUDA Toolkit version
        ≠
PyTorch CUDA runtime version
```

Compatibility matters more than identical version numbers.

---

### 3. Verify the NVIDIA driver before troubleshooting PyTorch

Start with:

```bash
nvidia-smi
```

If the driver itself is broken, reinstalling PyTorch will not fix the driver.

---

### 4. A CUDA Toolkit installation is not always required

If you are only running pre-built PyTorch binaries:

```text
NVIDIA Driver ✅
PyTorch CUDA ✅
CUDA Toolkit ⭕
```

If you are developing or compiling CUDA software:

```text
NVIDIA Driver ✅
CUDA Toolkit ✅
PyTorch CUDA ✅
```

---

### 5. Keep each software layer understandable

```text
Hardware
   ↓
Driver
   ↓
CUDA runtime
   ↓
PyTorch
   ↓
Your ML application
```

This makes troubleshooting significantly easier.

---

### 6. Use virtual environments

Windows:

```powershell
python -m venv .venv
.venv\Scripts\activate
```

Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

---

### 7. Use the official PyTorch installation selector

🔗 https://pytorch.org/get-started/locally/

PyTorch regularly changes:

* Supported CUDA versions
* Python versions
* GPU architectures
* Wheel builds
* Platform support

Therefore, installation commands may need updating over time.

---

### 8. Prefer distribution-supported Linux drivers

Ubuntu:

```bash
ubuntu-drivers install
```

Debian:

```text
Use Debian's NVIDIA packages
```

Fedora:

```text
Use Fedora/RPM Fusion-supported packages
```

Arch:

```text
Use Arch-supported NVIDIA packages
```

Avoid mixing multiple independent driver installation methods.

---

### 9. Save your working environment

After successfully configuring PyTorch:

```bash
python -m pip freeze > requirements.txt
```

This makes it easier to recreate the Python environment later.

---

### 10. Check the actual GPU during training

```bash
watch -n 1 nvidia-smi
```

This can help monitor:

```text
GPU utilization
GPU memory
temperature
running processes
```

---

# 🎉 CONCLUSION

A working NVIDIA + PyTorch environment is best understood as several independent layers.

## Windows

```text
NVIDIA GPU
    ↓
NVIDIA Driver
    ↓
Optional CUDA Toolkit
    ↓
Python virtual environment
    ↓
PyTorch CUDA build
    ↓
GPU computation
```

## Linux

```text
NVIDIA GPU
    ↓
Distribution NVIDIA Driver
    ↓
Optional CUDA Toolkit
    ↓
Python virtual environment
    ↓
PyTorch CUDA build
    ↓
GPU computation
```

The most important verification sequence is:

```text
1. GPU detected
      ↓
2. NVIDIA driver works
      ↓
3. CUDA Toolkit works (if installed)
      ↓
4. Correct PyTorch build installed
      ↓
5. torch.cuda.is_available()
      ↓
6. Actual CUDA computation succeeds
```

If these checks pass, your system is ready for GPU-accelerated deep learning with PyTorch.

---

# 💬 FEEDBACK

If you encounter an issue, please include the output of:

```bash
nvidia-smi
```

```bash
nvcc --version
```

```bash
python --version
```

and:

```python
import torch

print(torch.__version__)
print(torch.version.cuda)
print(torch.cuda.is_available())
print(torch.cuda.device_count())

if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))
```

This makes troubleshooting much easier.

Feel free to:

* Open an issue
* Fork the repository
* Submit a pull request
* Suggest improvements to this documentation

---

# 📥 CLONE THIS REPOSITORY

Clone the repository:

```bash
git clone https://github.com/djayepro3/CUDA-PyTorch-VSCode-GPU-Setup.git
```

Navigate to the project directory:

```bash
cd CUDA-PyTorch-VSCode-GPU-Setup
```

---

<p align="center">

### 🚀 Happy Coding and GPU Training!

**CUDA + PyTorch + VS Code**

</p>
