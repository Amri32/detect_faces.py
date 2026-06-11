# Quick Start Guide

<cite>
**Referenced Files in This Document**
- [detect_faces.py](file://detect_faces.py)
- [requirements.txt](file://requirements.txt)
</cite>

## Table of Contents
1. [Introduction](#introduction)
2. [Prerequisites](#prerequisites)
3. [Installation](#installation)
4. [Basic Usage](#basic-usage)
5. [Essential Parameters](#essential-parameters)
6. [Typical Workflows](#typical-workflows)
7. [Expected Output Structure](#expected-output-structure)
8. [Step-by-Step Tutorials](#step-by-step-tutorials)
9. [Common Command Variations](#common-command-variations)
10. [Troubleshooting Guide](#troubleshooting-guide)
11. [Next Steps for Advanced Usage](#next-steps-for-advanced-usage)

## Introduction
CaptureFace is a Python-based tool designed to automatically detect faces in photos and videos. It scans a folder of media files, applies OpenCV Haar Cascade detection, saves cropped face images, and generates a summary report. The tool supports both local folders and Google Drive shared folders, making it easy to process media stored remotely.

## Prerequisites
Before using CaptureFace, ensure you have:
- Python 3.7 or higher installed on your system
- Basic familiarity with command-line interfaces
- Access to a folder containing photos and/or videos (local or Google Drive)

## Installation
Install the required dependencies using pip:
- Install packages listed in requirements.txt

**Section sources**
- [requirements.txt:1-6](file://requirements.txt#L1-L6)

## Basic Usage
The simplest way to run CaptureFace is to specify an input folder and an output folder. The tool will:
- Scan the input folder recursively for supported image and video files
- Detect faces using OpenCV Haar Cascades
- Save cropped face images and annotated versions
- Generate a summary CSV and console output

Example command:
- Run detection on a local folder and save results to an output directory

**Section sources**
- [detect_faces.py:10-13](file://detect_faces.py#L10-L13)
- [detect_faces.py:291-447](file://detect_faces.py#L291-L447)

## Essential Parameters
CaptureFace accepts several command-line arguments to customize behavior:

- Input source selection
  - Local folder input: specify the path to a folder containing photos and/or videos
  - Google Drive input: provide a shared folder URL or ID; files will be downloaded to a temporary folder before processing

- Output destination
  - Specify the directory where results will be saved (default: ./output)

- Detection tuning
  - Scale factor: adjusts the image pyramid scaling for face detection
  - Minimum neighbors: controls sensitivity to reduce false positives
  - Minimum face size: sets the smallest face size in pixels

- Video processing
  - Sample rate: processes every N-th frame in videos to balance speed and coverage

**Section sources**
- [detect_faces.py:291-346](file://detect_faces.py#L291-L346)

## Typical Workflows
CaptureFace supports two primary workflows:

1. Local folder processing
   - Prepare a folder containing photos and/or videos
   - Run the detection script pointing to your input folder
   - Review results in the output directory

2. Google Drive processing
   - Obtain a shared folder URL or ID from Google Drive
   - Run the detection script with the Google Drive option
   - The tool will download files to a temporary location, process them, and clean up afterward

**Section sources**
- [detect_faces.py:354-365](file://detect_faces.py#L354-L365)
- [detect_faces.py:351-358](file://detect_faces.py#L351-L358)

## Expected Output Structure
After processing, CaptureFace creates the following output structure:
- Root output directory
  - faces/ subdirectory containing cropped face images grouped by source file
  - annotated/ subdirectory containing original images with detected faces highlighted
  - summary.csv file with per-file statistics

The summary CSV includes columns for file name, type, number of faces detected, number of face crops saved, and any errors encountered during processing.

**Section sources**
- [detect_faces.py:199-222](file://detect_faces.py#L199-L222)
- [detect_faces.py:243-286](file://detect_faces.py#L243-L286)
- [detect_faces.py:412-418](file://detect_faces.py#L412-L418)

## Step-by-Step Tutorials

### Tutorial 1: First-Time Setup and Local Folder Processing
1. Prepare your media files
   - Create a folder containing photos and/or videos you want to analyze
   - Ensure the folder contains supported file types (images: JPG, PNG, BMP, WebP, TIFF; videos: MP4, AVI, MOV, MKV, WMV, FLV, WebM)

2. Install dependencies
   - Install the required packages using pip

3. Run the detection
   - Execute the script with your input and output directories specified

4. Review results
   - Check the output directory for cropped faces, annotated images, and the summary CSV

**Section sources**
- [detect_faces.py:376-384](file://detect_faces.py#L376-L384)
- [detect_faces.py:412-418](file://detect_faces.py#L412-L418)

### Tutorial 2: Google Drive Integration
1. Share your Google Drive folder
   - Make the folder containing your media files publicly accessible or share it with the appropriate permissions

2. Get the folder URL or ID
   - Copy the shared folder URL or extract the folder ID from the URL

3. Run the detection with Google Drive
   - Execute the script with the Google Drive option and your chosen output directory

4. Monitor progress
   - The tool will download files to a temporary location, process them, and clean up afterward

**Section sources**
- [detect_faces.py:302-310](file://detect_faces.py#L302-L310)
- [detect_faces.py:354-358](file://detect_faces.py#L354-L358)

## Common Command Variations
CaptureFace supports various command combinations to suit different needs:

- Basic usage with local input
  - Run detection on a local folder with default settings

- Google Drive with custom output
  - Process files from a shared Google Drive folder and save results to a custom output directory

- Tuned detection parameters
  - Adjust scale factor, minimum neighbors, and minimum face size for different scenarios

- Video sampling control
  - Modify the sample rate to process more or fewer frames in videos

**Section sources**
- [detect_faces.py:10-13](file://detect_faces.py#L10-L13)
- [detect_faces.py:317-345](file://detect_faces.py#L317-L345)

## Troubleshooting Guide

### Common Issues and Solutions

1. Missing dependencies
   - Problem: ImportError or ModuleNotFoundError when running the script
   - Solution: Install all required packages using pip

2. Invalid input folder
   - Problem: Error indicating the input folder does not exist
   - Solution: Verify the path to your input folder and ensure it exists

3. Empty input folder
   - Problem: Warning that no images or videos were found
   - Solution: Add supported media files to your input folder or adjust the folder path

4. Google Drive access issues
   - Problem: Failure to download files from Google Drive
   - Solution: Ensure the shared folder is accessible and the URL/ID is correct

5. Memory or performance issues with large videos
   - Problem: Slow processing or memory constraints with long videos
   - Solution: Increase the sample rate to process fewer frames, or split the video into smaller segments

**Section sources**
- [detect_faces.py:363-365](file://detect_faces.py#L363-L365)
- [detect_faces.py:386-390](file://detect_faces.py#L386-L390)
- [detect_faces.py:354-358](file://detect_faces.py#L354-L358)

## Next Steps for Advanced Usage
Once comfortable with basic usage, consider these advanced configurations:

- Fine-tuning detection parameters
  - Experiment with different scale factors and minimum neighbors to optimize detection accuracy for your specific use case

- Batch processing strategies
  - Organize media files into subfolders for easier batch processing and result management

- Integration with other tools
  - Use the generated CSV for downstream analysis or integrate with other computer vision workflows

- Performance optimization
  - Adjust sample rates for videos based on your quality requirements and processing time constraints

- Custom output formatting
  - Modify the script to change output formats or add additional metadata extraction

**Section sources**
- [detect_faces.py:317-345](file://detect_faces.py#L317-L345)
- [detect_faces.py:412-418](file://detect_faces.py#L412-L418)