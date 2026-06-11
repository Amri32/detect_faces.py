# Installation and Setup

<cite>
**Referenced Files in This Document**
- [requirements.txt](file://requirements.txt)
- [detect_faces.py](file://detect_faces.py)
- [.gitignore](file://.gitignore)
</cite>

## Table of Contents
1. [Introduction](#introduction)
2. [System Requirements](#system-requirements)
3. [Virtual Environment Setup](#virtual-environment-setup)
4. [Dependency Installation](#dependency-installation)
5. [Operating System-Specific Setup](#operating-system-specific-setup)
6. [Verification Steps](#verification-steps)
7. [Troubleshooting Guide](#troubleshooting-guide)
8. [Environment Configuration Recommendations](#environment-configuration-recommendations)
9. [Conclusion](#conclusion)

## Introduction
CaptureFace is a Python-based face detection and cropping tool designed to scan folders containing photos and videos, detect faces using OpenCV Haar Cascades combined with the supervision library, save cropped face images, and report face counts per file and in total. It supports processing local directories and Google Drive shared folders or files.

Key capabilities:
- Face detection in images and videos
- Cropping detected faces into individual images
- Annotated output images with bounding boxes and labels
- CSV summary reporting
- Optional Google Drive integration for remote datasets

## System Requirements
- Python 3.x (recommended: Python 3.8 or newer)
- Operating systems:
  - Windows (7 SP1 or later)
  - macOS (10.14 or later)
  - Linux (Ubuntu 18.04 or later; other distributions supported)

Notes:
- The project uses OpenCV for image/video processing and relies on system-level libraries for video codec support on Linux.
- Ensure sufficient disk space for input media and output directories.

**Section sources**
- [detect_faces.py:10-14](file://detect_faces.py#L10-L14)
- [detect_faces.py:291-346](file://detect_faces.py#L291-L346)

## Virtual Environment Setup
Best practices:
- Create a dedicated virtual environment for CaptureFace to isolate dependencies.
- Use Python's built-in venv module or conda/miniconda environments.
- Activate the environment before installing dependencies.

Recommended commands:
- Create environment: python -m venv captureface_env
- Activate environment:
  - Windows: captureface_env\Scripts\activate
  - macOS/Linux: source captureface_env/bin/activate
- Verify Python version: python --version

Why use virtual environments:
- Prevents conflicts with system packages
- Ensures reproducible installations
- Simplifies dependency management

**Section sources**
- [.gitignore:6-7](file://.gitignore#L6-L7)

## Dependency Installation
Dependencies are defined in the requirements file and installed via pip. The project requires:
- supervision (minimum version specified)
- opencv-python (minimum version specified)
- numpy (minimum version specified)
- tqdm (minimum version specified)
- gdown (minimum version specified)

Installation steps:
1. Navigate to the project root directory
2. Activate your virtual environment
3. Install dependencies: pip install -r requirements.txt

Post-installation verification:
- Confirm successful installation by importing required modules in Python
- Run the main script with --help to verify CLI availability

**Section sources**
- [requirements.txt:1-6](file://requirements.txt#L1-L6)

## Operating System-Specific Setup

### Windows
Prerequisites:
- Python 3.x installed and added to PATH
- Visual C++ Redistributable (if not already present)
- Administrative privileges for package installation (if required)

Setup steps:
1. Open Command Prompt or PowerShell as Administrator
2. Create and activate a virtual environment
3. Install dependencies using pip
4. Test installation by running: python detect_faces.py --help

Common considerations:
- Ensure Python architecture (32-bit vs 64-bit) matches your system
- If encountering build errors during installation, upgrade pip, setuptools, and wheel: pip install --upgrade pip setuptools wheel

### macOS
Prerequisites:
- Xcode Command Line Tools installed (xcode-select --install)
- Python 3.x via Homebrew or official installer

Setup steps:
1. Open Terminal
2. Create and activate a virtual environment
3. Install dependencies using pip
4. If OpenCV fails to install, install OpenCV via Homebrew first, then reinstall opencv-python via pip

macOS-specific tips:
- For Intel Macs: Standard pip installation usually works
- For Apple Silicon Macs: Use Python from python.org or Homebrew to avoid architecture mismatches

### Linux
Prerequisites:
- GCC and build-essential packages
- Python 3.x development headers
- System video codecs for OpenCV video support

Ubuntu/Debian:
1. Update package manager: sudo apt update
2. Install system dependencies:
   - sudo apt install build-essential python3-dev python3-venv
   - Additional video codecs: sudo apt install ffmpeg libsm6 libxext6 libxrender-dev libglib2.0-0 libxvidcore-dev libx264-dev
3. Create and activate a virtual environment
4. Install dependencies using pip

Other distributions:
- Fedora/RHEL: sudo dnf install gcc python3-devel python3-venv
- Arch/Manjaro: sudo pacman -S gcc python3 python3-venv

Linux-specific notes:
- If OpenCV installation fails, install system packages for video codecs before pip installation
- Some distributions require additional Qt libraries for GUI components

**Section sources**
- [detect_faces.py:291-346](file://detect_faces.py#L291-L346)

## Verification Steps
After installation, verify your setup with these checks:

1. Basic CLI verification:
   - python detect_faces.py --help
   - Should display usage information and available options

2. Module import test:
   - python -c "import cv2; import numpy; import supervision; import gdown; print('All modules imported successfully')"

3. Simple processing test:
   - Create a small test directory with one or two images
   - Run: python detect_faces.py --input ./test_input --output ./test_output
   - Verify that output images and summary.csv are generated

4. Google Drive integration test (optional):
   - Use a small public Google Drive folder with test media
   - Run: python detect_faces.py --google-drive YOUR_DRIVE_URL_OR_ID --output ./test_output

Expected outcomes:
- Successful processing without errors
- Output directory contains:
  - faces/ subdirectory with cropped face images
  - annotated/ subdirectory with labeled images
  - summary.csv with processing results
- Console output shows processing progress and final statistics

**Section sources**
- [detect_faces.py:412-428](file://detect_faces.py#L412-L428)
- [detect_faces.py:445-447](file://detect_faces.py#L445-L447)

## Troubleshooting Guide

### Common Installation Issues

#### pip Installation Failures
Symptoms:
- Build errors during opencv-python installation
- "Microsoft Visual C++ 14.0 is required" on Windows
- "command 'xcrun' failed" on macOS

Solutions:
1. Upgrade pip, setuptools, and wheel:
   - pip install --upgrade pip setuptools wheel
2. Windows-specific:
   - Install Microsoft C++ Build Tools
   - Use pre-compiled wheels: pip install --only-binary=all opencv-python
3. macOS-specific:
   - Install Xcode Command Line Tools: xcode-select --install
   - Consider using conda-forge channel: conda install -c conda-forge opencv-python
4. Linux-specific:
   - Install system dependencies for video codecs
   - Use distribution-specific package managers when pip fails

#### Dependency Conflicts
Symptoms:
- Version mismatch errors
- Import failures for specific modules

Solutions:
1. Clean installation:
   - pip uninstall supervision opencv-python numpy tqdm gdown
   - pip cache purge
   - pip install -r requirements.txt
2. Use compatible versions:
   - Check requirements.txt for minimum versions
   - Consider using conda environments for stricter dependency resolution

#### OpenCV Issues
Symptoms:
- ImportError: libopencv_python
- Video processing failures
- Missing codec errors

Solutions:
1. Reinstall opencv-python:
   - pip uninstall opencv-python
   - pip install opencv-python-headless
2. Install system video codecs:
   - Windows: Install K-Lite Codec Pack
   - macOS: brew install ffmpeg
   - Linux: sudo apt install ffmpeg

#### Permission Errors
Symptoms:
- Permission denied when writing to output directory
- Access errors during Google Drive downloads

Solutions:
1. Run with appropriate permissions
2. Use absolute paths for input and output directories
3. Ensure write permissions for the target output directory

#### Memory Issues
Symptoms:
- Out of memory errors during video processing
- Slow performance on large videos

Solutions:
1. Increase sample rate for video processing (--sample-rate)
2. Process smaller batches of videos
3. Close other memory-intensive applications

**Section sources**
- [detect_faces.py:237-239](file://detect_faces.py#L237-L239)
- [detect_faces.py:191-194](file://detect_faces.py#L191-L194)

## Environment Configuration Recommendations

### Recommended Python Versions
- Primary support: Python 3.8+
- Compatibility: Python 3.7+ (with potential limitations)
- Avoid: Python 2.x or very old Python 3.x versions

### Virtual Environment Best Practices
- Always use isolated environments for projects
- Pin dependency versions in requirements.txt
- Document environment setup steps
- Consider using requirements-dev.txt for development dependencies

### Directory Structure
- Input directory: Place photos and videos to process
- Output directory: Will contain results (created automatically)
- Temporary directories: Managed automatically for Google Drive downloads

### Performance Tuning
- Adjust --sample-rate for video processing speed vs accuracy trade-off
- Modify --min-neighbors for sensitivity tuning
- Use headless mode for server environments without GUI

### Security Considerations
- When using Google Drive integration, review shared folder permissions
- Limit access to sensitive input directories
- Regularly update dependencies to address security vulnerabilities

**Section sources**
- [detect_faces.py:317-345](file://detect_faces.py#L317-L345)
- [detect_faces.py:354-357](file://detect_faces.py#L354-L357)

## Conclusion
CaptureFace provides a straightforward installation process with clear system requirements and dependency management. By following the virtual environment best practices and OS-specific setup instructions, you can reliably deploy the tool across different platforms. The verification steps and troubleshooting guide should help resolve most common installation issues. For production deployments, consider the environment configuration recommendations to ensure optimal performance and security.

Key takeaways:
- Use virtual environments for isolation
- Install system dependencies before Python packages when needed
- Test with small datasets before processing large collections
- Monitor resource usage during video processing
- Keep dependencies updated for security and compatibility