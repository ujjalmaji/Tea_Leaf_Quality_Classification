#  Tea Leaf Quality Classifier for Robotic Plucking

##  Introduction
Automated robotic tea plucking requires a highly accurate, real-time vision system to identify optimal harvesting targets: **young, healthy tea leaves free of disease or pest damage**. [cite_start]Traditional image-based approaches often struggle in real-world agricultural settings due to visual ambiguity, high foliage density, and overlapping leaves[cite: 177, 178]. 

To overcome these challenges, this project implements a **Two-Stage Detection and Classification Pipeline**:
1. [cite_start]**Localization (Stage 1):** A YOLOv8 object detection model identifies and isolates individual tea leaves within dense, complex foliage backgrounds[cite: 58, 349]. 
2. **Fine-Grained Classification (Stage 2):** A custom deep learning architecture (`IterationVIT`) evaluates the localized regions of interest (ROIs) to confirm the leaf's health and readiness for plucking. 

---

##  System Architecture 

### Stage 1: Leaf Localization via YOLOv8
[cite_start]Instead of analyzing the entire image indiscriminately, the system uses object detection to focus solely on relevant leaf regions[cite: 350]. [cite_start]YOLOv8 is utilized for this stage due to its high detection accuracy, real-time inference capability, and suitability for resource-constrained robotic environments[cite: 351, 352]. 
* **Input:** High-resolution camera feed from the robotic arm.
* [cite_start]**Process:** The single-stage detector performs localization in a single forward pass, generating bounding boxes around potential target leaves[cite: 232, 233].
* **Output:** Cropped bounding boxes (Regions of Interest) containing individual leaves.

### Stage 2: Health & Quality Classification via IterationVIT
Once a leaf is localized, it must be inspected for disease and pest damage. We utilize an **Iterative Region of Interest Encoding Transformer (IterationVIT)**, which combines convolutional feature extraction with transformer-based sequence modeling.
* **Bottleneck Convolution:** The cropped ROI first passes through a `BottleneckConv` module, which utilizes 1x1 and 3x3 convolutions with Batch Normalization and ReLU activations to extract foundational visual features.
* **Iterative ROI Encoding:** The features are flattened and passed into an `IterativeROI` module, where a 4-layer Transformer Encoder processes the patches alongside positional embeddings.
* **Output:** A multi-class prediction determining if the leaf is a "Young Healthy Leaf" or if it exhibits specific pest/disease symptoms. 

---

##  Technical Implementation

### The Classification Backbone (IterationVIT)
The core classifier leverages a hybrid CNN-Transformer architecture built in PyTorch. Below is the structural implementation for the Stage 2 classification model.

```python
import torch
import torch.nn as nn
from einops import rearrange

# 1. Bottleneck Convolution Module for feature extraction
class BottleneckConv(nn.Module):
    def __init__(self, in_channels, bottleneck_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, bottleneck_channels, kernel_size=1),
            nn.BatchNorm2d(bottleneck_channels),
            nn.ReLU(),
            nn.Conv2d(bottleneck_channels, bottleneck_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(bottleneck_channels),
            nn.ReLU(),
            nn.Conv2d(bottleneck_channels, in_channels, kernel_size=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU()
        )

    def forward(self, x):
        return self.block(x)

# 2. Iterative ROI Encoding Module using Transformers
class IterativeROI(nn.Module):
    def __init__(self, num_iters, embed_dim, num_patches):
        super().__init__()
        self.num_iters = num_iters
        self.linear = nn.Linear(2, embed_dim)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(embed_dim, nhead=8, dim_feedforward=512),
            num_layers=4
        )
        self.num_patches = num_patches

    def forward(self, x):
        B, C, H, W = x.shape
        out = rearrange(x, 'b c h w -> b (h w) c')
        pos = torch.rand(B, self.num_patches**2, 2).to(x.device)
        pos_embed = self.linear(pos)
        tokens = out[:, :self.num_patches**2, :] + pos_embed
        tokens = self.transformer(tokens)
        return tokens

# 3. Full Stage-2 Classification Model
class IterationVIT(nn.Module):
    def __init__(self, num_classes=2): # e.g., 0: Diseased, 1: Healthy Young Leaf
        super().__init__()
        self.conv = BottleneckConv(3, 64)
        self.roi_encoder = IterativeROI(num_iters=3, embed_dim=64, num_patches=16)
        self.classifier = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.conv(x)
        x = self.roi_encoder(x)
        x = x.mean(dim=1)
        return self.classifier(x)
```

### The 2-Stage Pipeline Execution
The robotic plucking system bridges localization and classification iteratively:

```python
from ultralytics import YOLO

# Load Stage 1 (Detection) and Stage 2 (Classification)
detector = YOLO('yolov8_foliage_weights.pt')
classifier = IterationVIT(num_classes=2)
classifier.load_state_dict(torch.load('iteration_vit_weights.pth'))
classifier.eval()

def process_camera_feed(frame):
    # Stage 1: Detect all leaves in the dense foliage
    results = detector(frame)
    
    plucking_targets = []
    
    for box in results[0].boxes:
        # Extract Region of Interest
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        leaf_roi = frame[y1:y2, x1:x2]
        
        # Preprocess ROI for IterationVIT
        tensor_roi = preprocess_for_vit(leaf_roi)
        
        # Stage 2: Classify Health and Quality
        with torch.no_grad():
            prediction = classifier(tensor_roi)
            predicted_class = torch.argmax(prediction, dim=1).item()
            
            # If classified as a Healthy Young Leaf (Target == 1)
            if predicted_class == 1:
                plucking_targets.append((x1, y1, x2, y2))
                
    return plucking_targets
```

---

##  Future Enhancements (Multimodal Integration)
To further improve robotic decision-making in ambiguous lighting or complex environmental conditions, this architecture can be expanded into a multimodal framework. [cite_start]By capturing metadata from the YOLOv8 detections (such as bounding box area ratios and spatial layouts), the system can generate structured textual severity descriptions[cite: 363, 408]. [cite_start]Fusing these contextual text embeddings (via BERT) with the visual embeddings (via ResNet/IterationVIT) can yield a highly robust, explainable system for agricultural decision-making[cite: 423, 466, 467].
