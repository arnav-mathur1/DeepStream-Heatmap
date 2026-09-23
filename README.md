# Detection Heatmap in NVIDIA DeepStream

This was a smaller side-project I worked on as a part of my work in Summer 2026. It is a CUDA-accelerated detection heatmap overlay for an NVIDIA DeepStream object detection pipeline.

In this repo, I hope to describe what it does and things I had to consider.

# Overview

This heatmap overlay  accumulates RF-DETR detection locations over time, highlighting high-density regions directly on the video stream.

For each frame, detections produced by the DeepStream inference pipeline contribute to a heatmap. Repeated detections in the same region increase the heatmap intensity, while older detections gradually fade.

This makes it possible to visualize where an object detector is firing most frequently over time.

# How it Works

The heatmap module attaches a probe to the DeepStream video pipeline and reads the NvDsBatchMeta object metadata associated with each frame.

For every detection, it:
- Takes the center of the detected bounding box and its confidence score.
- Adds a Gaussian-shaped contribution around that location to a persistent GPU heatmap.
- Multiplies the existing heatmap by a decay factor so older detections gradually fade.
- Alpha-blends the heatmap directly onto the video frame using CUDA.

The accumulator remains on the GPU between frames, so a location becomes more prominent when the detector repeatedly fires there rather than simply displaying the detections from the current frame.

# Demo

This [demo](https://drive.google.com/file/d/12U2Fu3Z3mltrYqR--QqEXjLtvlITXfLh/view?usp=sharing) uses an earlier version of the RF-DETR UAV detector with a relatively high false-positive rate.

This provides a useful visualization of the heatmap behavior: locations where the detector repeatedly produces detections develop persistent hotspots, while isolated detections gradually disappear.

# How to Run

The heatmap is intended to be integrated into an existing NVIDIA DeepStream detection pipeline.

To use it:
- Build heatmap_overlay.cu as part of the DeepStream application or as a shared library.
- Attach heatmap_overlay_attach() to the desired GStreamer/DeepStream pad after detection metadata is available.
- Enable the overlay through heatmap_overlay.ini or the corresponding environment variables.
- Run the normal DeepStream application.

Example configuration:

```text
[heatmap_overlay]
enable=1
source_id=0
decay=0.985
sigma_px=28
alpha_max=0.65
splat=0.55
```

The main parameters control how quickly old detections fade, how broadly each detection contributes to the heatmap, and how strongly the resulting heatmap is blended onto the video.

## Files

```text
heatmap_overlay.cu
heatmap_overlay.h
```

`heatmap_overlay.cu` contains the CUDA heatmap generation and video overlay implementation.

`heatmap_overlay.h` exposes the heatmap integration interface used by the DeepStream pipeline.

