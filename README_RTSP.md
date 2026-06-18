# RTSP Streaming & Looping Implementation Guide

This guide explains the architectural and code changes implemented in the DINOv3 DeepStream application to support robust, continuous RTSP video streaming and looping.

---

## 🛠️ Modifications & Architectural Changes

### 1. Headless RTSP Delivery
Instead of rendering visualization outputs to local EGL/X11 display windows (which fails in headless Docker environments), we integrated a GStreamer RTSP Server (`GstRTSPServer`) and converted the visualization channels to parallel H.264 RTP output streams:
* **Frame Transcoding**: Visualization streams are converted from their native layout (`RGBA` or `NV12`) to `I420` format, hardware-encoded to H.264 via `nvv4l2h264enc` (inserting SPS/PPS headers), and payloaded to RTP packets via `rtph264pay`.
* **UDP Relay**: The payloaded RTP packets are pushed locally to a UDP socket using `udpsink` (ports `5400` to `5404`).
* **RTSP Mounting**: A GLib main loop manages the RTSP server instance, capturing the RTP packages from the local UDP sockets and publishing them as RTSP streams.

### 2. Synchronization & Client Disconnection Fixes
To prevent video players (like VLC, FFplay, and mpv) from freezing or failing to connect at arbitrary times, we configured several critical properties:
* **Periodic Header Delivery (`config-interval=1`)**: Added `config-interval=1` to the `rtph264pay` payloader. This forces the sender to output the H.264 Sequence Parameter Set (SPS) and Picture Parameter Set (PPS) headers every second, allowing players to join a stream immediately without waiting for a keyframe.
* **Monotonic Timestamps (`do-timestamp=true`)**: Configured the RTSP server's internal receiving `udpsrc` with `do-timestamp=true`. This forces the server to assign fresh timestamps based on its own running clock, shielding clients from the time-reset discontinuity when the input file loops.
* **Rate Control (`sync=true`)**: Enabled sync pacing on the output `udpsink` when processing video files. This locks the processing speed to the natural frame rate of the video (e.g., 30 FPS) to prevent GPU over-utilization and network flooding.

### 3. Fail-Safe Continuous File Looping
We addressed GStreamer pipeline termination when processing video files:
* **GLib Main Loop Integration**: Upgraded the execution cycle to run within a GLib main loop (`GMainLoop`), ensuring callbacks are processed asynchronously without blocking.
* **Manual Flush Seeking**: Designed a `BusCallData` structure to pass both the main loop and pipeline reference to the bus callback function. Upon catching a GStreamer End-of-Stream (`GST_MESSAGE_EOS`) event on the bus, we trigger a manual flush-seek back to the start of the stream (`gst_element_seek`), allowing infinite continuous looping of the input video files.

---

## ⚙️ Configuration Parameters

### GStreamer Port Mapping
The application streams each inference task's visualizer to its own local UDP port, which is picked up and exposed by the RTSP server on port `554`:

| Task Head | local UDP port | RTSP Mount Path | RTSP URL |
| :--- | :---: | :--- | :--- |
| **Main Stream (Tiled)** | `5400` | `/ds-test` | `rtsp://<host_ip>:554/ds-test` |
| **Depth Estimation** | `5401` | `/depth` | `rtsp://<host_ip>:554/depth` |
| **Object Detection** | `5402` | `/detection` | `rtsp://<host_ip>:554/detection` |
| **Segmentation** | `5403` | `/segmentation` | `rtsp://<host_ip>:554/segmentation` |
| **Optical Flow** | `5404` | `/optical-flow` | `rtsp://<host_ip>:554/optical-flow` |

### Command Line Flags
The following RTSP-specific parameters were added to the C++ application config:
* `--rtsp-output`: Set to `true` (default) to stream via RTSP, or `false` to attempt local EGL rendering.
* `--rtsp-port`: RTSP server port (default: `554`).
* `--rtsp-mount`: Main mount point path (default: `/ds-test`).

---

## 💻 Source Modifications

1. **[app_config.h](dinov3_deepstream/src/config/app_config.h)**: Added RTSP server parameters (`rtsp_output`, `rtsp_port`, `rtsp_mount`) to the configuration schema.
2. **[pipeline_builder.cpp](dinov3_deepstream/src/pipeline/pipeline_builder.cpp)**: Refactored visualization and inference branches to support `udpsink` targets and individual head `tee` outputs.
3. **[main.cpp](dinov3_deepstream/src/main.cpp)**: Replaced the application loop with `GMainLoop`, handled manual seek on `GST_MESSAGE_EOS`, and initialized the `GstRTSPServer` with multiple media factories.
4. **[CMakeLists.txt](dinov3_deepstream/CMakeLists.txt)** & **[Dockerfile](Dockerfile)**: Configured dependencies and compilation links for `libgstrtspserver-1.0`.
