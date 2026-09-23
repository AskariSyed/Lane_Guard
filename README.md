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

![Classical Computer Vision Preprocessing Stages](assets/classical/03_preprocessing_stages.png)
*Figure 1: Deterministic feature engineering stages of the classical CV pipeline: HLS color thresholding, Canny edge detection, trapezoidal ROI extraction, and Hough accumulator voting.*

![Classical Lane Departure Warning HUD Output](assets/classical/11_departure_warning_hud.png)
*Figure 2: Fitted linear highway boundaries, synthesized green drivable corridor, and Heads-Up Display (HUD) indicating lateral offset and departure warning status.*

### Learned Pipelines

LaneGuard implements and evaluates two complementary deep-learning segmentation models fine-tuned under an identical training protocol:

#### 1. YOLOv8n-seg Baseline
Implemented in [`laneguard_yolo.ipynb`](laneguard_yolo.ipynb):
- **Architecture**: Anchor-free detection backbone with C2f feature aggregation, decoupled detection/segmentation head, and non-maximum suppression (85 fused layers, 3,258,454 parameters, 11.3 GFLOPs at $640 \times 640$).
- **Classes**: Class 0 (`lane_line` ribbons) and Class 1 (`drivable_area` derived corridor polygon).
- **Post-Processing**: Masks resized to $1280 \times 720$; bottom near-bumper pixels ($y > 0.70 \cdot H$) partition left and right lane boundaries relative to vehicle center ($x_{\text{veh}} = 640\text{ px}$). If a boundary is absent, a heuristic fallback offset of $420\text{ px}$ is applied.

![YOLOv8 Segmentation Predictions Grid](assets/yolo/yolov8_segmentation_results_grid.png)
*Figure 3: Semantic mask predictions and HUD outputs produced by fine-tuned YOLOv8n-seg across representative highway scenes.*

#### 2. YOLO26n-seg Modern Baseline
Implemented in [`LaneGuard_YOLO26n_Segmentation.ipynb`](LaneGuard_YOLO26n_Segmentation.ipynb):
- **Architecture**: Modern end-to-end NMS-free dual-branch architecture incorporating C3k2 feature extractors and C2PSA spatial attention blocks (136 fused layers / 309 unfused layers, 2,689,274 parameters, 9.1 GFLOPs at $640 \times 640$).
- **Design Advantage**: 17.5% fewer parameters and 19.5% lower computational complexity compared to YOLOv8n-seg, with native dual-label assignment eliminating non-maximum suppression latency variance.
- **Harmonized Logic**: Implements identical data transformation, spatial error measurement ($y = 680\text{ px}$), LDWS thresholding ($\pm 80\text{ px}$), and metric export.

---

## Experimental Setup

| Parameter | Classical Pipeline | Learned YOLOv8-Seg Baseline | Learned YOLO26n-seg Baseline |
| :--- | :--- | :--- | :--- |
| **Notebook Implementation** | [`laneguard.ipynb`](laneguard.ipynb) | [`laneguard_yolo.ipynb`](laneguard_yolo.ipynb) | [`LaneGuard_YOLO26n_Segmentation.ipynb`](LaneGuard_YOLO26n_Segmentation.ipynb) |
| **Model Checkpoint** | Algorithmic (OpenCV 4.x) | `yolov8n-seg.pt` (fused 3.26M params, 6.8 MB) | `yolo26n-seg.pt` (fused 2.69M params, 6.4 MB) |
| **Detection Head** | Deterministic grouping | Standard Anchor-Free NMS Head | Native End-to-End NMS-Free Head |
| **Training Budget** | None (Rule-based) | 15 epochs, batch 16, 600 frames | 15 epochs, batch 16, 600 frames |
| **Input Resolution** | $1280 \times 720$ | $640 \times 640$ (network) $\rightarrow$ $1280 \times 720$ (HUD) | $640 \times 640$ (network) $\rightarrow$ $1280 \times 720$ (HUD) |
| **Hardware** | Host CPU (Single Core) | NVIDIA Tesla T4 GPU (16 GB VRAM) | NVIDIA Tesla T4 GPU (16 GB VRAM) |
| **Timing Metric** | Wall-clock `time.perf_counter()` | Synchronized CUDA Timing (`torch.cuda.synchronize()`) | Synchronized CUDA Timing (`torch.cuda.synchronize()`) |
| **Evaluation Set** | 6 qualitative scenes | 2,782 frames (TuSimple test set, 0% leakage) | 2,782 frames (TuSimple test set, 0% leakage) |

---

## Quantitative Results

### 1. Validation Set Segmentation Performance (150 Images)

Validation metrics were recorded on the 150 held-out validation images (710 target instances) at epoch 15 for both learned architectures:

| Target Class | Metric | YOLOv8n-seg | YOLO26n-seg | Architectural Difference |
| :--- | :--- | :---: | :---: | :--- |
| **Overall (All Classes)** | **Box Precision** | **0.953** | 0.946 | High localization precision across both models |
| | **Box Recall** | **0.932** | 0.915 | YOLOv8 recovers slightly more bounding anchors |
| | **Box mAP@0.50** | **0.963** | 0.957 | Comparable coarse bounding box performance |
| | **Box mAP@0.50:0.95** | **0.828** | 0.823 | High tight-IoU bounding performance |
| | **Mask Precision** | **0.647** | 0.641 | Pixel-level boundary precision matched within 0.6% |
| | **Mask Recall** | **0.632** | 0.618 | YOLOv8 achieves 1.4% higher mask coverage |
| | **Mask mAP@0.50** | **0.544** | 0.537 | Coarse segmentation threshold comparable |
| | **Mask mAP@0.50:0.95** | **0.369** | 0.359 | Strict mask IoU metric matched within 1.0% |
| **`lane_line` (Class 0)** | Mask mAP@0.50 | 0.260 | **0.274** | YOLO26 achieves +1.4% higher ribbon mAP |
| | Mask mAP@0.50:0.95 | 0.0567 | **0.0606** | YOLO26 spatial attention slightly improves tight ribbon IoU |
| **`drivable_area` (Class 1)** | Mask mAP@0.50 | **0.828** | 0.801 | YOLOv8 captures broad drivable corridor slightly better |
| | Mask mAP@0.50:0.95 | **0.681** | 0.657 | YOLOv8 achieves higher continuous corridor overlap |

> **Evaluation Honesty Note**: The validation split metrics evaluate overlap against project-derived polygon targets and originate from the frame-level split. As detailed below, the sequence-level benchmark provides the leakage-free empirical proof.

The low mask metrics for `lane_line` across both models (mAP@0.50 of ~0.26–0.27) stem from two structural factors:
1. **Narrow Ribbon IoU Penalty**: Road markings are narrow ribbons ($\pm 6\text{ px}$ width). A spatial displacement of only 2–3 pixels penalizes the Intersection over Union (IoU) calculation heavily.
2. **Limited Training Budget**: Both models were trained for only 15 epochs on 600 frames. Loss trajectories showed box loss and segmentation loss still descending.

---

### 2. Full Dataset Evaluation & Cross-Model Benchmark (2,782 Independent Sequences)

To establish rigorous empirical proof free from temporal data leakage, both models were evaluated against **all 2,782 official TuSimple test sequences** (`test_label.json`), ensuring **0% frame or sequence overlap** with training data.

Evaluations were executed using the exact same evaluation harness, identical test manifests (`results/evaluation_manifest.csv`), and identical near-bumper spatial error scanlines ($y = 680\text{ px}$).

![Empirical Benchmark: YOLOv8n-seg vs YOLO26n-seg across 2,782 TuSimple Clips](assets/comparison/yolov8_vs_yolo26_full_comparison.png)
*Figure 4: Empirical cross-model evaluation across 2,782 independent TuSimple driving sequences. (Top-Left) Spatial pixel error distributions for boundary and center estimates at near-bumper scanline $y = 680\text{ px}$. (Top-Right) Per-clip Mean Absolute Error distribution across driving clips. (Bottom-Left) Synchronized CUDA runtime latency distributions on NVIDIA Tesla T4. (Bottom-Right) Detection failure rate comparison.*

#### Empirical Performance Summary

All metrics below represent actual measured values recorded on Kaggle using an NVIDIA Tesla T4 GPU (also saved in [`results/yolov8_vs_yolo26_summary.csv`](results/yolov8_vs_yolo26_summary.csv)):

| Evaluation Metric | YOLOv8n-seg Baseline | YOLO26n-seg Modern Baseline | Empirical Finding & Architectural Rationale |
| :--- | :---:| :---:| :--- |
| **Evaluated Sequences / Frames** | 2,782 clips / 2,782 frames | 2,782 clips / 2,782 frames | Identical standardized evaluation manifest |
| **Test Sequence Partition** | `test_label.json` (Official) | `test_label.json` (Official) | **0% clip leakage** with training set |
| **Spatial Evaluation Line** | $y = 680\text{ px}$ (Near Bumper) | $y = 680\text{ px}$ (Near Bumper) | Evaluates where lateral vehicle offset is critical |
| **Lane Boundary MAE** | **21.42 px** | 22.44 px | **Boundary Parity**: Models agree within 1.02 px (~4.8%) |
| **Lane Boundary RMSE** | **23.39 px** | 23.59 px | Negligible 0.20 px difference in boundary outliers |
| **Lane Boundary Median AE** | **21.00 px** | 22.50 px | Median boundary error differs by only 1.50 px |
| **Lane Center MAE** | **37.62 px** | 70.06 px | YOLOv8 achieves lower center error due to dual-boundary detection |
| **Lane Center RMSE** | **57.13 px** | 82.66 px | YOLO26 impacted by single-boundary fixed fallback |
| **Lane Center Median AE** | **10.50 px** | 87.50 px | Centering precision tightly clustered for YOLOv8 |
| **Dual Boundary Detection Rate** | **48.6%** (1,351 frames) | 16.4% (457 frames) | YOLOv8 recovers both left/right boundaries 2.96× more often |
| **Single Boundary Detection Rate** | 44.9% (1,248 frames) | 53.9% (1,500 frames) | YOLO26 frequently triggers single-line fallback heuristic |
| **Detection Failure Rate** | **6.58%** (183 frames) | 29.65% (825 frames) | Deeper dual-branch head required more epochs to converge |
| **Mean Inference Latency** | **10.95 ms** | 14.40 ms | Synchronized CUDA stream benchmark on Tesla T4 |
| **Median Inference Latency** | **10.81 ms** | 14.19 ms | Low jitter across all 2,782 frames |
| **P95 Latency** | **11.93 ms** | 15.76 ms | Predictable worst-case latency under 16 ms |
| **Inference Throughput** | **91.4 FPS** | 69.5 FPS | **Both exceed 60 FPS** real-time driving threshold |
| **Fused Model Parameters** | 3,258,454 (100%) | **2,689,274 (82.5%)** | **YOLO26 saves 17.5% parameters** (569k fewer weights) |
| **Computational Complexity** | 11.3 GFLOPs (100%) | **9.1 GFLOPs (80.5%)** | **YOLO26 saves 19.5% FLOPs** (2.2 GFLOPs lower) |
| **Stripped Weights Size** | 6.8 MB | **6.5 MB** | Lightweight deployment footprint |

#### In-Depth Proof & Distribution Analysis

1. **Boundary Precision Equivalence (Panel 1: Top-Left)**:
   - The boundary error distributions for both models are nearly identical: YOLOv8 achieves an MAE of $21.42\text{ px}$ while YOLO26 achieves $22.44\text{ px}$.
   - The boxplots in Figure 4 show narrow interquartile ranges (IQR) with median boundary errors of $21.0\text{ px}$ and $22.5\text{ px}$ respectively. This confirms that both networks locate physical highway lane markings with high spatial consistency when markings are detected.

2. **Per-Clip Error Tracking and Center Error Disparity (Panel 2: Top-Right)**:
   - For lane center estimation, YOLOv8 achieves a median absolute error of only $10.50\text{ px}$ (mean $37.62\text{ px}$), whereas YOLO26 exhibits a median of $87.50\text{ px}$ (mean $70.06\text{ px}$).
   - This difference is directly linked to boundary completeness: in 48.6% of frames, YOLOv8 segments both the left and right lane boundaries simultaneously, allowing the lane center to be calculated directly from bilateral boundary medians ($x_{\text{center}} = (x_L + x_R)/2$).
   - In contrast, YOLO26 detects both boundaries in only 16.4% of frames and falls back to single-boundary fixed-offset estimation ($x_L + 420\text{ px}$ or $x_R - 420\text{ px}$) in 53.9% of frames. On curving highway segments or wide lanes, a static $420\text{ px}$ half-width introduces systematic offset deviations.

3. **Loss Architecture & Training Convergence (Panel 4: Bottom-Right)**:
   - Under the constrained budget of 15 epochs on 600 training images, YOLOv8 converged faster because its standard decoupled head uses standard NMS with well-established anchor assignments.
   - YOLO26 introduces an end-to-end NMS-free dual-branch architecture combining L1 loss, classification loss, and mask loss across 309 layers. At epoch 15, YOLO26's training logs show box loss ($0.72$) and segmentation loss ($0.61$) still actively decreasing. As a result, boundary confidence scores on ambiguous frames fell below the $0.25$ threshold, producing a higher failure rate ($29.65\%$ vs $6.58\%$). Given a 50–100 epoch schedule, YOLO26's dual-branch head is expected to converge to higher dual-boundary recall.

4. **Runtime Efficiency and Complexity (Panel 3: Bottom-Left)**:
   - Latency distributions measured via CUDA synchronization (`torch.cuda.synchronize()`) across all 2,782 frames show tight, single-modal distributions without long tail stalls.
   - YOLOv8 executes at $10.95\text{ ms}$ (91.4 FPS) and YOLO26 executes at $14.40\text{ ms}$ (69.5 FPS).
   - While YOLO26 has more layers (136 fused layers vs 85 for YOLOv8), it is more computationally efficient, reducing parameter count by **17.5%** ($2.69\text{M}$ vs $3.26\text{M}$) and FLOPs by **19.5%** ($9.1\text{ GFLOPs}$ vs $11.3\text{ GFLOPs}$).

---

### 3. Realistic Classical-CV Profiling

To compare deep-learning inference with classical image processing, the deterministic OpenCV pipeline was profiled across 200 iterations on real TuSimple highway scenes ($1280 \times 720$):

| Test Condition / Scene | Mean Latency | Median Latency | Std Deviation | Throughput | Execution Hardware |
| :--- | :---:| :---:| :---:| :---:| :--- |
| **Typical Highway Frame** (`01_ground_truth_lanes.png`) | 25.71 ms | 25.86 ms | 2.52 ms | 38.9 FPS | Host CPU (Single Core) |
| **Vehicle Scene with Multiple Markings** (`02_numbered_ground_truth.png`) | 31.81 ms | 25.11 ms | 13.57 ms | 31.4 FPS | Host CPU (Single Core) |
| **Dense Edge / Surface Clutter** (`06_roi_masked_edges.png`) | 70.70 ms | 68.42 ms | 10.41 ms | 14.1 FPS | Host CPU (Single Core) |
| **Overall Realistic Frame Average** (4 scenes, 200 iterations) | 49.25 ms | 55.72 ms | 22.62 ms | 20.3 FPS | Host CPU (Single Core) |
| **Historical Synthetic Blank Frame** (Empty black array) | 10.60 ms | 10.55 ms | 0.45 ms | 94.3 FPS | Host CPU (Bypasses Hough/polyfit) |

> **Hardware Context**: Classical CV executes sequentially on CPU cores, where latency varies by $3\times$ depending on edge density and texture clutter. Learned neural networks execute on GPU tensor cores with constant execution time independent of image content clutter.

---

## Qualitative Case Analysis

The perception pipelines were evaluated across representative highway geometries from the TuSimple test sequences:

![Classical vs YOLOv8 Side-by-Side Comparison Grid](assets/comparison/comparison_full_grid.png)
*Figure 5: Side-by-side qualitative comparison of Classical CV (left column) and YOLOv8n-seg (right column) across 6 representative highway driving scenes.*

![YOLO26 Qualitative Scene Evaluation](assets/yolo26/yolo26_qualitative_scenes.png)
*Figure 6: YOLO26n-seg semantic segmentation, boundary delineation, drivable corridor overlay, and HUD warning status across representative test scenes.*

### Cross-Pipeline Per-Scene Evaluation Summary

All models were evaluated under the harmonized heuristic departure warning threshold ($\pm 80.0\text{ px}$):

| Scene | Geometry / Challenge | Classical CV Offset | Classical LDWS ($\pm 80\text{ px}$) | YOLOv8 Offset | YOLOv8 LDWS ($\pm 80\text{ px}$) | Comparative Qualitative Finding |
| :--- | :--- | :---:| :---:| :---:| :---:| :--- |
| **Scene 1** (`20.jpg`) | Rightward highway curve | -25.20 px | `CENTERED` | -118.00 px | `WARNING: RIGHT` | Linear fit averages across entire trapezoid, dampening curve drift; learned masks track bumper-level curvature |
| **Scene 2** (`16.jpg`) | Curving highway with lead vehicle | -68.49 px | `CENTERED` | -135.25 px | `WARNING: RIGHT` | Classical linear fit underestimates lateral displacement near the bumper |
| **Scene 3** (`3.jpg`)  | Straight open highway | +50.74 px | `CENTERED` | +27.00 px | `LANE CENTERED` | **Full Agreement**: Both pipelines maintain stable centered state on tangent highway |
| **Scene 4** (`8.jpg`)  | Clear dashed lane stripes | -34.95 px | `CENTERED` | -8.50 px | `LANE CENTERED` | **Full Agreement**: Both pipelines track dashed striping without departure warning |
| **Scene 5** (`15.jpg`) | Vehicle drifting rightward | -84.00 px | `WARNING: RIGHT` | -82.00 px | `WARNING: RIGHT` | **Perfect Warning Concordance**: Both pipelines detect near-identical offset (-84 px vs -82 px) and trigger rightward warnings |
| **Scene 6** (`2.jpg`)  | Subtle highway curvature | +14.82 px | `CENTERED` | -125.00 px | `WARNING: RIGHT` | Classical linear grouping fails to capture curve trajectory, mistaking curved boundary for vertical tangent |

---

## Lane-Departure Warning Logic

The Lane Departure Warning System (LDWS) is a heuristic implementation operating in image pixel space:

1. **Vehicle Center**: Assumed to correspond to the horizontal midpoint at the bottom edge of the image:
   $$x_{\text{vehicle}} = \frac{W}{2} = 640\text{ px}$$
2. **Lane Center**:
   - **Classical CV**: Evaluated at the bottom frame row ($y = 720\text{ px}$) from the fitted boundary lines:
     $$x_{\text{lane center}} = \frac{x_{\text{left}}(720) + x_{\text{right}}(720)}{2}$$
   - **YOLOv8-Seg**: Evaluated as the midpoint of median horizontal coordinates of Class 0 pixels sampled near the bumper ($y \in [0.70 \cdot H, H]$):
     $$x_{\text{lane center}} = \frac{\text{median}(x_{\text{left}}) + \text{median}(x_{\text{right}})}{2}$$
3. **Lateral Offset**:
   $$\Delta x = x_{\text{vehicle}} - x_{\text{lane center}}$$
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
2. **[`laneguard_yolo.ipynb`](laneguard_yolo.ipynb)**: Converts TuSimple polylines to YOLO segmentation format, fine-tunes `yolov8n-seg`, evaluates validation metrics, and includes full dataset evaluation across all relevant TuSimple clips.
3. **[`LaneGuard_YOLO26n_Segmentation.ipynb`](LaneGuard_YOLO26n_Segmentation.ipynb)**: Parallel learned segmentation baseline implementing `yolo26n-seg`, executing the identical training protocol, full dataset evaluation, and cross-model comparison.
4. **[`laneguard_comparison.ipynb`](laneguard_comparison.ipynb)**: Runs the comparative analysis and benchmark routines across both pipelines.
5. **[`laneguard_conclusion.ipynb`](laneguard_conclusion.ipynb)**: Standalone summary notebook replicating the comparative results.

### Full Dataset Evaluation & Cross-Model Benchmarking

LaneGuard includes a standardized, shared evaluation protocol implemented identically across [`laneguard_yolo.ipynb`](laneguard_yolo.ipynb) and [`LaneGuard_YOLO26n_Segmentation.ipynb`](LaneGuard_YOLO26n_Segmentation.ipynb):
- **Unified Evaluation Manifest (`results/evaluation_manifest.csv`)**: Programmatically discovers all relevant TuSimple evaluation clips/frames, strictly separating them from training frames to ensure both models evaluate the exact same sequences.
- **Ground-Truth Spatial Error Evaluation**: Measures boundary-level and lane-center Mean Absolute Error (MAE), Root Mean Square Error (RMSE), and median error at lower near-bumper scanlines ($y = 680\text{ px}$).
- **Per-Clip Analysis**: Aggregates valid predictions, detection failure rates, and tracking accuracy per individual driving sequence.
- **Runtime Profiling**: Comprehensive latency benchmark across the entire evaluation set with CUDA stream synchronization (mean, median, standard deviation, P95 latency, and FPS).
- **Automated Cross-Model Comparison**: Exports results to standardized CSVs (`results/yolov8_full_evaluation.csv`, `results/yolov26_full_evaluation.csv`) and generates direct comparative tables and distribution plots.

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
