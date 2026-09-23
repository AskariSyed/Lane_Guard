# LaneGuard: Comparative Study of Classical Computer Vision and Learned Segmentation for Monocular Lane Perception

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green.svg)](https://opencv.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange.svg)](https://pytorch.org/)
[![Ultralytics YOLOv8](https://img.shields.io/badge/YOLOv8--Seg-Ultralytics-blueviolet.svg)](https://github.com/ultralytics/ultralytics)
[![Dataset: TuSimple](https://img.shields.io/badge/Dataset-TuSimple%20Benchmark-yellow.svg)](https://github.com/TuSimple/tusimple-benchmark)
[![License: MIT](https://img.shields.io/badge/License-MIT-lightgrey.svg)](LICENSE)

An applied computer-vision proof-of-concept evaluating the trade-offs between an algorithmic edge-and-line pipeline and a lightweight deep-learning segmentation model for monocular highway lane perception and heuristic lane-departure warning.

---

## Research Question

How do classical edge- and geometry-based lane detection pipelines compare with lightweight learned segmentation models in terms of implementation complexity, inference latency, qualitative behavior across highway road geometries, and suitability for heuristic lane-departure warning when evaluated on forward-facing monocular imagery?

---

## Project Scope

LaneGuard is an undergraduate-level computer-vision prototype designed to examine the operational differences between hand-crafted geometric vision rules and learned semantic masks. 

The implementation operates strictly in two-dimensional image pixel space on individual monocular video frames. It does not perform metric camera calibration, Inverse Perspective Mapping (IPM) to Bird's-Eye-View (BEV), vehicle state estimation, or CAN bus interfacing. It is not an autonomous driving system, not a production ADAS solution, and carries no functional safety certification under ISO 26262 or related automotive standards.

---

## Dataset

All experiments use highway imagery from the [TuSimple Lane Detection Benchmark](https://github.com/TuSimple/tusimple-benchmark). 

TuSimple sequences consist of forward-facing camera video clips ($1280 \times 720$ resolution) recorded under predominantly dry, clear-weather daytime highway driving conditions.

### Annotation Format

Ground-truth lane markings in TuSimple are annotated as polylines defined across predefined vertical scanlines:
- `h_samples`: Discrete vertical pixel coordinates ($y \in [160, 710]$ at $10\text{ px}$ intervals).
- `lanes`: Horizontal pixel coordinates ($x$) corresponding to each height, where $-2$ denotes an unannotated or occluded scanline.

---

## Dataset and Label Construction

TuSimple supplies lane polylines rather than semantic or instance segmentation masks. Training masks for the learned pipeline were generated programmatically from these polyline coordinates:

1. **Lane Line Ribbons (`lane_line`)**:
   - Discrete $(x, y)$ polyline points for each annotated lane were dilated horizontally by $\pm 6\text{ px}$ (`thickness=6`) to form closed ribbon polygons.
   - Coordinates were normalized to $[0, 1]$ for the YOLO segmentation format.
2. **Derived Corridor Masks (`drivable_area`)**:
   - The two innermost lane boundaries closest to the horizontal center of the frame ($x = 640\text{ px}$) were identified.
   - A closed polygon was constructed between these two lines across their shared vertical height span.

> **Methodological Clarification on Labels**: TuSimple provides lane marking annotations. The corridor masks used for the segmentation experiment are derived directly from these lane coordinates. They must be interpreted as derived lane-corridor targets rather than independently annotated drivable-area ground truth. They do not represent curbs, road shoulders, or unstriped boundaries, and they do not account for dynamic obstacles.

---

## Dataset Split and Temporal Leakage Limitation

The dataset was constructed from the 3,626 annotated frames available in the three TuSimple training annotation files (`label_data_0601.json`, `label_data_0313.json`, `label_data_0531.json`):

- **Training Set**: 600 frames (`images/train`, `labels/train`)
- **Validation Set**: 150 frames (`images/val`, `labels/val`), containing 710 ground-truth target instances (560 `lane_line` instances, 150 `drivable_area` instances)
- **Qualitative Case Set**: 6 test frames sampled at uniform file index intervals across the test sequences.

> **Important Experimental Limitation (Temporal Data Leakage)**: In the original experiment, frames were partitioned using uniform random shuffling at the individual frame level (`np.random.seed(42)`). Because TuSimple sequences are organized as continuous 20-frame video clips recorded at 1-second intervals, random frame-level shuffling distributes temporally adjacent frames from the same driving sequence across both the training and validation subsets. Consequently, the reported validation metrics evaluate interpolation on familiar scenes and should not be interpreted as a leakage-free estimate of sequence-level out-of-distribution generalization.

---

## Methods

### Classical Pipeline

Implemented in [`laneguard.ipynb`](laneguard.ipynb), the classical pipeline relies on deterministic image processing without trainable parameters:

1. **HLS Color Space Filtering**:
   - Converts the input RGB frame to HLS color space.
   - White markings: $H \in [0, 255]$, $L \in [185, 255]$, $S \in [0, 255]$.
   - Yellow markings: $H \in [10, 40]$, $L \in [60, 255]$, $S \in [60, 255]$.
   - A bitwise OR combines the masks to preserve marking pixels under daylight contrast.
2. **Gaussian Smoothing and Canny Edge Detection**:
   - Grayscale conversion followed by a $5 \times 5$ Gaussian blur kernel.
   - Canny edge detection with hysteresis thresholds $T_{\text{low}} = 50$, $T_{\text{high}} = 150$.
   - Combined with the HLS mask via bitwise OR.
3. **Trapezoidal Region of Interest (ROI)**:
   - Masks the upper visual field on $1280 \times 720$ frames:
     - Bottom vertices: $(0, 720)$ and $(1280, 720)$
     - Top vertices: $(768, 360)$ and $(512, 360)$
4. **Probabilistic Hough Transform**:
   - Line segment detection using `cv2.HoughLinesP` ($\rho = 1\text{ px}$, $\theta = \pi / 180$, accumulator threshold $= 35$, minimum line length $= 30\text{ px}$, maximum line gap $= 80\text{ px}$).
5. **Slope Partitioning and Outlier Filtering**:
   - Calculates segment slope as $m = \Delta x / \Delta y$. Segments with $|\Delta y| < 15\text{ px}$ or $|m| < 0.25$ or $|m| > 2.5$ are rejected.
   - Segments are partitioned into left candidates ($x_{\text{mid}} < 0.52 \cdot W$, $m < 0$) and right candidates ($x_{\text{mid}} > 0.48 \cdot W$, $m > 0$).
   - The median slope $m_{\text{med}}$ is computed per candidate pool; segments deviating by $|m - m_{\text{med}}| \ge 0.35$ are discarded.
6. **First-Order Linear Fitting**:
   - Inlier segment endpoints are fitted to a first-order linear model ($x = my + b$) using least squares (`np.polyfit(y, x, 1)`).
   - If one boundary is missing, `complete_single_lane` synthesizes the opposing line using assumed highway perspective dimensions ($850\text{ px}$ lane width at the bottom, $280\text{ px}$ at the horizon).
   - *Exploratory Prototype Note*: A standalone second-order polynomial routine (`polynomial_lane_fit`) is implemented in `laneguard.ipynb` Cell 38, but it was not integrated into `detect_lanes` or used in the comparative evaluations.
7. **Temporal Smoothing**:
   - A moving-average buffer (`LaneSmoother`) maintains a FIFO queue of up to 8 recent boundary fits (`deque(maxlen=8)`), averaging coefficients across consecutive video frames.

### Learned Pipeline

Implemented in [`laneguard_yolo.ipynb`](laneguard_yolo.ipynb), this pipeline applies a lightweight neural network to predict binary-style semantic masks:

1. **Architecture and Target Representation**:
   - A `yolov8n-seg` architecture (85 layers, 3,258,454 parameters, 11.3 GFLOPs at $640 \times 640$) was fine-tuned from pre-trained weights.
   - The network was repurposed for two semantic-style targets:
     - **Class 0 (`lane_line`)**: All visible road boundary ribbons. Individual lane markings are not assigned distinct instance identities.
     - **Class 1 (`drivable_area`)**: The derived corridor polygon between the innermost lane lines.
2. **Inference and Feature Post-Processing**:
   - Inference is executed at a confidence threshold of $0.25$.
   - Segmented masks are resized back to $1280 \times 720$.
   - Left and right boundary assignment is performed during post-processing: Class 0 mask pixels in the near-bumper region ($y > 0.70 \cdot H$) are partitioned relative to the vehicle horizontal center ($x_{\text{veh}} = W / 2 = 640\text{ px}$).
   - The lane center is estimated from the median horizontal positions of these pixels:
     $$x_{\text{lane\_center}} = \frac{\text{median}(x_{\text{left\_pts}}) + \text{median}(x_{\text{right\_pts}})}{2}$$
   - If only one boundary produces mask pixels, a fixed half-width offset heuristic ($420\text{ px}$) is applied ($x_{\text{left}} + 420$ or $x_{\text{right}} - 420$).

---

## Experimental Setup

| Parameter | Classical Pipeline | Learned YOLOv8-Seg Pipeline |
| :--- | :--- | :--- |
| **Notebook Implementation** | [`laneguard.ipynb`](laneguard.ipynb) | [`laneguard_yolo.ipynb`](laneguard_yolo.ipynb) |
| **Model Type** | Algorithmic (OpenCV 4.x) | `yolov8n-seg.pt` (3.26M parameters, 6.8 MB) |
| **Training Budget** | None (Rule-based) | 15 epochs, batch size 16, 600 training frames |
| **Evaluated Fitting Order** | First-order linear ($x = my + b$) | Pixel-level semantic mask sampling |
| **Input Resolution** | $1280 \times 720$ | $640 \times 640$ (network input) $\rightarrow$ $1280 \times 720$ (HUD) |
| **Hardware** | Host CPU (Single Core) | NVIDIA Tesla T4 GPU (16 GB VRAM) |
| **Timing Metric** | Wall-clock `time.perf_counter()` | Wall-clock `time.perf_counter()` (no CUDA events) |
| **Dataset Split** | Evaluated on test frames | Frame-level random split (600 train / 150 val) |

---

## Quantitative Results

### Segmentation Metrics

Validation metrics were recorded on the 150 held-out validation images (710 target instances) at epoch 15:

| Target Class | Instances | Box P | Box R | Box mAP@0.50 | Box mAP@0.50:0.95 | Mask P | Mask R | Mask mAP@0.50 | Mask mAP@0.50:0.95 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Overall (All Classes)** | 710 | 0.953 | 0.932 | 0.963 | 0.828 | 0.647 | 0.632 | 0.544 | 0.369 |
| **`lane_line` (Class 0)** | 560 | 0.982 | 0.959 | 0.987 | 0.784 | 0.458 | 0.446 | 0.260 | 0.0567 |
| **`drivable_area` (Class 1)** | 150 | 0.925 | 0.906 | 0.940 | 0.872 | 0.836 | 0.818 | 0.828 | 0.681 |

> **Evaluation Honesty Note**: The reported validation metrics come from the original frame-level random split and therefore should not be interpreted as a leakage-free estimate of sequence-level generalization. Furthermore, this evaluation assesses overlap against project-derived polygon targets; it does not represent official TuSimple benchmark polyline accuracy.

The low mask metrics for `lane_line` (mAP@0.50 of 0.260 and mAP@0.50:0.95 of 0.0567) are attributable to two known experimental factors:
1. **Ribbon IoU Sensitivity**: Lane line targets are narrow ribbons ($\pm 6\text{ px}$ width). A spatial displacement of only a few pixels significantly reduces the Intersection over Union (IoU) calculation.
2. **Limited Training Budget**: The network was fine-tuned for only 15 epochs on 600 frames. Training curves show that box loss (0.65) and segmentation loss (0.64) were still decreasing, indicating the model had not reached convergence. Conversely, the larger continuous area of the `drivable_area` class achieved a Mask mAP@0.50 of 0.828.

---

### Runtime Benchmarking

#### 1. Historical Notebook Benchmark (Synthetic Black Frames)
In [`laneguard_comparison.ipynb`](laneguard_comparison.ipynb) and [`laneguard_conclusion.ipynb`](laneguard_conclusion.ipynb), runtime was measured over 50 iterations using blank black dummy arrays (`np.zeros((720, 1280, 3))`):

| Pipeline | Execution Hardware | Mean Latency | Throughput | Scope / Methodological Caveat |
| :--- | :--- | :---:| :---:| :--- |
| **Classical CV** | Host CPU | 10.60–10.77 ms | 92.8–94.3 FPS | **Artificially deflated**: Zero edges/lines found; Hough grouping and polyfit are bypassed |
| **YOLOv8-Seg (Inference)** | NVIDIA Tesla T4 GPU | 9.51 ms | 105.1 FPS | Forward pass only on blank frame (`predict`) |
| **YOLOv8-Seg (End-to-End)** | NVIDIA Tesla T4 GPU | 13.44–13.58 ms | 73.6–74.4 FPS | Forward pass + mask resize + pixel offset + HUD |

#### 2. Realistic Classical-CV Profiling (Measured on Highway Frames)
To evaluate the true computational cost of edge extraction, Hough accumulator voting, slope partitioning, and linear fitting, Classical CV was re-benchmarked across 200 iterations on real TuSimple highway scenes ($1280 \times 720$):

| Test Condition / Scene | Mean Latency | Median Latency | Std Deviation | Throughput |
| :--- | :---:| :---:| :---:| :---:|
| **Typical Highway Frame** (`01_ground_truth_lanes.png`) | 25.71 ms | 25.86 ms | 2.52 ms | 38.9 FPS |
| **Vehicle Scene with Multiple Markings** (`02_numbered_ground_truth.png`) | 31.81 ms | 25.11 ms | 13.57 ms | 31.4 FPS |
| **Dense Edge / Surface Clutter** (`06_roi_masked_edges.png`) | 70.70 ms | 68.42 ms | 10.41 ms | 14.1 FPS |
| **Overall Realistic Frame Average** (4 scenes, 200 iterations) | 49.25 ms | 55.72 ms | 22.62 ms | 20.3 FPS |

> **Comparative Timing Caveat**: These measurements characterize different execution environments and workloads and should not be interpreted as a controlled CPU-vs-GPU performance comparison. The Classical CV pipeline was profiled on a host CPU executing OpenCV image operations, while YOLOv8-Seg was executed on a Tesla T4 GPU. Furthermore, GPU timing in the original notebooks was recorded via wall-clock `time.perf_counter()` rather than CUDA-event stream synchronization (`torch.cuda.Event`).

---

## Qualitative Case Analysis

The two pipelines were compared across 6 sample frames from the TuSimple test set in [`laneguard_comparison.ipynb`](laneguard_comparison.ipynb):

![Side-by-Side Comparison Grid](assets/comparison/comparison_full_grid.png)

### Per-Scene Evaluation Summary

Results evaluated under the harmonized heuristic threshold ($\pm 80.0\text{ px}$):

| Scene File | Classical Offset | Classical Status ($\pm 80\text{ px}$) | YOLO Offset | YOLO Status ($\pm 80\text{ px}$) | Case Analysis Observation |
| :--- | :---:| :---:| :---:| :---:| :--- |
| **Scene 1** (`20.jpg`) | -25.20 px | `CENTERED` | -118.00 px | `WARNING: DRIFTING RIGHT` | Linear fit averages across the trapezoid, dampening curve drift; YOLO tracks bumper-level curvature |
| **Scene 2** (`16.jpg`) | -68.49 px | `CENTERED` | -135.25 px | `WARNING: DRIFTING RIGHT` | Road curves rightward; linear fit underestimates lateral drift near the hood |
| **Scene 3** (`3.jpg`)  | +50.74 px | `CENTERED` | +27.00 px | `LANE CENTERED` | Straight highway; both pipelines indicate centered vehicle position |
| **Scene 4** (`8.jpg`)  | -34.95 px | `CENTERED` | -8.50 px | `LANE CENTERED` | Clear dashed stripes; both pipelines track within centering boundaries |
| **Scene 5** (`15.jpg`) | -84.00 px | `WARNING: DRIFTING RIGHT` | -82.00 px | `WARNING: DRIFTING RIGHT` | **Agreement**: Both pipelines detect near-identical offset; both trigger warnings under shared $\pm 80\text{ px}$ threshold |
| **Scene 6** (`2.jpg`)  | +14.82 px | `CENTERED` | -125.00 px | `WARNING: DRIFTING RIGHT` | Subtle road curve; classical linear grouping fails to capture curve trajectory |

*Note: Positive offset indicates the vehicle is to the left of the lane center; negative indicates right. These 6 samples represent qualitative case analysis and are not a statistically representative benchmark.*

---

## Lane-Departure Warning Logic

The Lane Departure Warning System (LDWS) is a heuristic implementation operating in image pixel space:

1. **Vehicle Center**: Assumed to correspond to the horizontal midpoint at the bottom edge of the image:
   $$x_{\text{vehicle}} = \frac{W}{2} = 640\text{ px}$$
2. **Lane Center**:
   - **Classical CV**: Evaluated at the bottom frame row ($y = 720\text{ px}$) from the fitted boundary lines:
     $$x_{\text{lane\_center}} = \frac{x_{\text{left}}(720) + x_{\text{right}}(720)}{2}$$
   - **YOLOv8-Seg**: Evaluated as the midpoint of median horizontal coordinates of Class 0 pixels sampled near the bumper ($y \in [0.70 \cdot H, H]$):
     $$x_{\text{lane\_center}} = \frac{\text{median}(x_{\text{left\_pts}}) + \text{median}(x_{\text{right\_pts}})}{2}$$
3. **Lateral Offset**:
   $$\Delta x = x_{\text{vehicle}} - x_{\text{lane\_center}}$$
4. **Harmonized Warning Logic**:
   - A single shared heuristic threshold of $\pm 80.0\text{ px}$ ($0.0625 \cdot W$) is applied:
     - $\Delta x > +80\text{ px}$: `WARNING: DRIFTING LEFT`
     - $\Delta x < -80\text{ px}$: `WARNING: DRIFTING RIGHT`
     - $|\Delta x| \le 80\text{ px}$: `CENTERED` / `LANE CENTERED`

> **Critical Note on the Warning Threshold**: The $\pm 80\text{ px}$ threshold is a project-specific pixel-space heuristic. It depends directly on camera mounting height, pitch, lens field-of-view, and image resolution ($1280 \times 720$). It is not an automotive safety standard (such as ISO 17361, which evaluates metric distance and time-to-line-crossing) and carries no safety certification.

---

## Failure Cases

Detailed inspection of the classical pipeline reveals several concrete operational failure modes:

1. **Curvature Underestimation from Linear Fitting**: First-order linear fitting ($x = my + b$) assumes zero road curvature. On curving highway segments (Scenes 1, 2, 6), slope averaging across the entire trapezoid ($y \in [360, 720]$) underestimates lateral vehicle displacement at the vehicle bumper.
2. **Static Extrapolation in Single-Lane Recovery**: The `complete_single_lane` routine uses static highway widths ($850\text{ px}$ at base, $280\text{ px}$ at horizon). If lane width deviates from this standard or if vehicle pitch changes, the synthesized opposing line introduces systematic lateral offset errors.
3. **Edge Sensitivity to Road Surface Clutter**: Pavement texture, tire skid marks, and shadow boundaries create extraneous Canny edges that distort the median slope calculation during Hough filtering.

---

## Limitations

1. **Pixel-Space Heuristic Without Metric Calibration**: All calculations operate in 2D image pixels without camera calibration (intrinsic matrix $K$, extrinsic pitch/roll/height) or Inverse Perspective Mapping (IPM) to Bird's-Eye-View (BEV). Pixel displacements do not scale linearly to physical meters.
2. **Temporal Data Leakage**: The YOLO model was evaluated on a frame-level random split rather than a sequence-partitioned split. Validation metrics should be interpreted with this leakage in mind.
3. **Derived Corridor Masks**: Corridor targets were constructed from lane polylines rather than true free-space annotations. The model cannot detect curbs, non-lane road boundaries, or obstacles.
4. **Limited Training Budget**: Fine-tuning for only 15 epochs on 600 frames was insufficient for boundary convergence on narrow lane lines, resulting in low mask mAP@0.50 (0.260).
5. **Absence of Production ADAS Architecture**: Real-world ADAS stacks rely on multi-sensor verification (radar, camera), dynamic temporal state estimation (Extended Kalman Filters, clothoid splines), vehicle dynamics models, and formal safety mechanisms. None of these components are implemented in this monocular prototype.
6. **Domain Constraint**: Evaluated exclusively on clear-weather daytime highway scenes from TuSimple. Performance under rain, snow, nighttime glare, and urban intersections is unverified.

---

## Reproducibility

### 1. Environment Setup
```bash
git clone https://github.com/AskariSyed/Lane_Guard.git
cd Lane_Guard

python -m venv venv
# Linux / macOS:
source venv/bin/activate
# Windows:
venv\Scripts\activate

pip install opencv-python numpy matplotlib pandas pyyaml ultralytics torch torchvision
```

### 2. Dataset Paths
The original notebooks were executed in a Kaggle environment referencing `/kaggle/input/...`. For local execution, configure the `TUSIMPLE_ROOT` environment variable or place the dataset in `./data/tusimple`:
```bash
export TUSIMPLE_ROOT="/path/to/tusimple"
```

### 3. Notebook Execution Order
1. **[`laneguard.ipynb`](laneguard.ipynb)**: Executes the classical CV pipeline (HLS, Canny, Hough, linear fit, LDWS logic).
2. **[`laneguard_yolo.ipynb`](laneguard_yolo.ipynb)**: Converts TuSimple polylines to YOLO segmentation format, fine-tunes `yolov8n-seg`, and evaluates validation metrics.
3. **[`laneguard_comparison.ipynb`](laneguard_comparison.ipynb)**: Runs the comparative analysis and benchmark routines across both pipelines.
4. **[`laneguard_conclusion.ipynb`](laneguard_conclusion.ipynb)**: Standalone summary notebook replicating the comparative results.

---

## My Contribution

I implemented and evaluated both perception pipelines in this repository:
- **Classical Pipeline**: Implemented HLS color filtering, Canny edge detection, trapezoidal ROI extraction, probabilistic Hough transform, slope filtering, median-slope outlier rejection, single-lane geometric synthesis, and temporal moving-average smoothing in Python and OpenCV ([`laneguard.ipynb`](laneguard.ipynb)).
- **Dataset Toolchain**: Developed data conversion scripts to parse TuSimple polylines, generate normalized polygon ribbons for lane markings, and derive closed drivable corridor polygons ([`laneguard_yolo.ipynb`](laneguard_yolo.ipynb)).
- **Model Fine-Tuning**: Configured and fine-tuned `yolov8n-seg` on the derived targets using a Tesla T4 GPU, evaluating precision, recall, and mAP across classes.
- **Telemetry and Benchmarking**: Implemented the lateral-offset estimation logic, HUD visualizer, and comparative benchmarking workflows ([`laneguard_comparison.ipynb`](laneguard_comparison.ipynb), [`laneguard_conclusion.ipynb`](laneguard_conclusion.ipynb)).

---

## Future Work

- **Sequence-Level Splitting**: Partition and retrain on strict clip-level boundaries to measure generalization free of temporal leakage.
- **Metric Calibration & IPM**: Calibrate camera intrinsics/extrinsics to map perception outputs into real-world meters in Bird's-Eye-View.
- **Temporal State Tracking**: Implement Kalman filtering or parametric clothoid spline tracking across consecutive frames to stabilize mask predictions.
- **ISO 17361 LDWS Evaluation**: Evaluate departure warnings using Time-to-Line-Crossing (TLC) and metric boundary thresholds rather than static pixel offsets.
- **Adverse Conditions**: Benchmark pipelines on night driving and rain datasets to evaluate domain transfer.

---

## References

1. **TuSimple Benchmark**: TuSimple Lane Detection Challenge. [https://github.com/TuSimple/tusimple-benchmark](https://github.com/TuSimple/tusimple-benchmark)
2. **YOLOv8**: Jocher, G., Chaurasia, A., & Qiu, J. (2023). *Ultralytics YOLO* (Version 8.0.0). [https://github.com/ultralytics/ultralytics](https://github.com/ultralytics/ultralytics)
3. **OpenCV**: Bradski, G. (2000). *The OpenCV Library*. Dr. Dobb's Journal of Software Tools.
4. **ISO 17361**: International Organization for Standardization. (2017). *Intelligent transport systems — Lane departure warning systems — Performance requirements and test procedures* (ISO Standard No. 17361:2017).
