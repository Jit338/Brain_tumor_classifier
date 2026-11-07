# Brain Tumor Classification System

This repository holds our end-to-end system for classifying brain tumors from MRI scans. The project is split into three main parts:
1. **Machine Learning Model** (Completed)
2. **Backend API** (In Progress)
3. **Frontend Dashboard** (In Progress)

---

## 1. Machine Learning Model (ResNet101)

This module contains the core deep learning pipeline that classifies brain tumors from 2D MRI slices. 

A major focus while building the model was preventing the data leakage issues common in public notebooks. Instead of blindly mixing all the 2D slices before splitting, we implemented a strict patient-level split. This forces the network to actually learn the tumor pathology instead of just memorizing a specific patient's skull shape.

### Dataset
We trained and evaluated the model entirely on the **Cheng Figshare Brain Tumor Dataset**. It contains 3,064 T1-weighted contrast-enhanced MRI scans from 233 patients, covering three specific tumor types: **Meningioma**, **Glioma**, and **Pituitary**. The raw data comes in `.mat` files.

### How it Works
- **Input:** A raw 2D MRI scan. The dataloader applies CLAHE to fix tissue contrast, resizes the image to 256x256, and converts it into a standard 3-channel RGB tensor.
- **Output:** The model predicts whether the scan shows a **Meningioma**, **Glioma**, or **Pituitary** tumor, outputting the final class probabilities via softmax.

### Architecture & Training Setup
- **Backbone:** PyTorch **ResNet101** (loaded with ImageNet1K_V1 weights). It is a heavy model, but the deep residual connections pull out complex medical textures reliably.
- **Custom Head:** Replaced the default 1000-class head with a custom Linear layer and Dropout (p=0.2) to keep overfitting in check.
- **Training Optimizations:**
  - **WSD Scheduler:** Standard step schedulers were causing validation loss spikes. We built a custom Warmup-Stable-Decay schedule using `SequentialLR` to let the weights settle during a long plateau before annealing the learning rate.
  - **SWA (Stochastic Weight Averaging):** Enabled for the final epochs. Averaging the weights pushed the model into a flatter, more stable minimum.
  - **Bilateral TTA:** During inference, the pipeline evaluates both the original image and a horizontally flipped version, averaging the probabilities. Rotation TTA was intentionally skipped because rotating an MRI messes up the absolute coordinates of brain anatomy (like the pituitary gland always sitting at the bottom center).

### Results & Metrics

Testing on the unseen patient split, the model hit **94.77% overall accuracy**.

#### Training vs Validation Loss
The validation loss stays flat and tracks the training loss closely. Lowering the classifier learning rate and dropout helped stabilize this curve.

<img width="536" height="456" alt="Screenshot 2026-11-08 001852" src="https://github.com/user-attachments/assets/0d182be5-3fd1-4e18-af61-6e3a5264c1d7" />

#### WSD Learning Rate Schedule
The custom Warmup-Stable-Decay schedule running over 25 epochs.

<img width="583" height="456" alt="Screenshot 2026-11-08 001901" src="https://github.com/user-attachments/assets/e65edf58-69ad-4a6a-be7f-10e4fef60d8a" />

#### Gradient Norms
Gradient clipping was set at 20.0 to stop exploding gradients. The average norm stayed healthy and dropped below 5.0 as the model converged.

<img width="542" height="451" alt="Screenshot 2026-11-08 001924" src="https://github.com/user-attachments/assets/ff975dff-21a1-41a8-8a67-c763fb0239ae" />

#### Final Classification Report
Glioma detection was the strongest across the three classes, hitting an F1-score of ~0.96.

<img width="721" height="327" alt="Screenshot 2026-11-08 001817" src="https://github.com/user-attachments/assets/578e4434-2d77-4142-a06d-eb7732d04532" />

#### Confusion Matrix
Out of 516 test slices, the model correctly identified 230 out of 241 Gliomas.

<img width="485" height="407" alt="Screenshot 2026-11-08 001833" src="https://github.com/user-attachments/assets/03b61a36-5d47-486e-9170-bc621d8118e3" />

---

## 2. Backend API (In Progress)

The backend service will wrap the trained ResNet101 weights into a REST API for real-time inference.

- **Planned Features:**
  - Endpoints to handle MRI image uploads and validation.
  - Automated preprocessing pipeline (CLAHE contrast enhancement and tensor normalization).
  - Model inference engine returning predicted tumor classes and softmax confidence scores in JSON format.

*(Setup instructions and API documentation will be updated once the backend is pushed.)*

---

## 3. Frontend Interface (In Progress)

A clean web interface for interacting with the classification model without touching the code.

- **Planned Features:**
  - Drag-and-drop uploader for 2D MRI scans.
  - Visual breakdown of the model's prediction and class probability distribution.
  - Side-by-side view of the raw MRI scan and the CLAHE-enhanced input.

*(Screenshots and local run commands will be added once the UI is ready.)*
