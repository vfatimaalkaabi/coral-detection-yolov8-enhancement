# Improving Underwater Coral Detection Using White Balance and CLAHE

CSBP441 – Applied Computer Vision Final Project, Fall 2026

This project evaluates whether image enhancement improves underwater coral detection and classification with YOLOv8n. We compare four pipelines on the same held-out test set:

1. Baseline YOLOv8n
2. Gray-world white balance + YOLOv8n
3. CLAHE + YOLOv8n
4. White balance + CLAHE + YOLOv8n

## Links

- [Master experiment notebook](https://colab.research.google.com/drive/1H8_SP3c7qteVTqQVi3kU9xG3YCX4H72q?usp=sharing)
- [Live demonstration notebook](https://colab.research.google.com/drive/1vPiTnGlELVAH_C8jWHm4pTXyULWy4TK3?usp=sharing)
- [Shared project files and model outputs](https://drive.google.com/drive/folders/13puEzl_o0rgRQ8pv4c54JHJqwjLhljUx?usp=share_link)
- [Roboflow Coral Detection Dataset, Version 2](https://universe.roboflow.com/data-sets-for-research/coral-detectionjh31e/dataset/2)

## Dataset

The original Roboflow Coral Detection Dataset contains 3,241 images and 38 classes. Three classes with no test-set annotations were excluded, producing a 35-class working dataset.

| Split | Images |
|---|---:|
| Training | 2,650 |
| Validation | 397 |
| Test | 169 |

Nine rare classes were oversampled in the training set only. Validation and test images, labels, and class mappings were unchanged.

## Experimental Settings

All four pipelines used:

- YOLOv8n with COCO-pretrained weights
- Image size: 640 × 640
- 50 epochs
- Batch size: 16
- AdamW optimizer, learning rate 0.001
- Random seed: 42
- NVIDIA Tesla T4 GPU in Google Colab
- The same fixed training, validation, and test splits

## Results

| Pipeline | Precision | Recall | mAP@50 | Latency per Image |
|---|---:|---:|---:|---:|
| Baseline | 0.323 | 0.421 | 0.402 | 14.8 ms |
| White balance | 0.390 | 0.455 | 0.439 | 28.8 ms |
| CLAHE | **0.411** | **0.467** | **0.480** | 26.0 ms |
| White balance + CLAHE | 0.411 | 0.408 | 0.438 | 40.7 ms |

CLAHE alone achieved the highest test mAP@50. The combined pipeline improved mAP@50 by 3.6 percentage points over the baseline, but increased latency by 175%; therefore, the hypothesis was partly supported.

## How to Run the Demo

1. Open the [live demonstration notebook](https://colab.research.google.com/drive/1vPiTnGlELVAH_C8jWHm4pTXyULWy4TK3?usp=sharing).
2. Select **Runtime → Change runtime type → T4 GPU**.
3. Run the setup cells from top to bottom.
4. Run each selected example cell to compare:
   - Baseline YOLOv8n prediction
   - White-balance + CLAHE image
   - Proposed pipeline prediction
5. If a live runtime fails, use the saved backup figures in the final cell.

The setup takes approximately 2–3 minutes. The selected examples run in under one minute after model loading.

## Project Contents

- `ACV_Project.ipynb` — master notebook for experiments, ablations, evaluation, and saved results
- `ACV_Demo.ipynb` — lightweight live demonstration notebook
- `report.pdf` — final report
- `presentation.pdf` — final presentation
- Google Drive folder — model weights, saved metrics, figures, and datasets

> The raw dataset is not stored in this repository. Use the Roboflow dataset link and shared Drive folder above.

## Software

Python 3, PyTorch, Ultralytics YOLOv8, OpenCV, NumPy, pandas, and Matplotlib.

Ultralytics version: `8.4.207`

## Attribution

The Coral Detection Dataset is provided through Roboflow Universe under CC BY 4.0. YOLOv8n and pretrained COCO weights are provided by Ultralytics. Gray-world white balance is based on Buchsbaum (1980), and CLAHE is based on Zuiderveld (1994).

ChatGPT and Claude were used for coding guidance, debugging support, and report drafting. The team reviewed, executed, and verified all generated code and experimental results.

## Team Contributions

| Member | Contribution |
|---|---|
| Mariam Alamoodi | Implemented and tuned CLAHE using validation data; created the CLAHE dataset; trained and evaluated the CLAHE-only model; measured computational cost and prepared related results. |
| Fatima Alkaabi | Prepared and checked the dataset; preserved the fixed splits; filtered unsupported classes; created the rare-class oversampling dataset; trained and evaluated the baseline YOLOv8n model; prepared baseline figures and results. |
| Amna Alneyadi | Implemented gray-world white balance; created the white-balance dataset; trained and evaluated the white-balance-only model; contributed to the shared evaluation procedure and report writing. |
| Meera Al Neyadi | Implemented the combined white-balance + CLAHE pipeline; trained and evaluated the combined model; prepared the live demo, qualitative examples, pipeline diagram, and final presentation materials.|
