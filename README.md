### Machine Learning Pipeline for Multi-Temporal Landsat Urban Classification

An automated satellite image processing and machine learning pipeline engineered in MATLAB. This project tracks and maps urban land modifications on the island of **Rhodes, Greece over a 35-year temporal span (1985 vs. 2020)** by training distinct Artificial Neural Networks (ANN) on multispectral Landsat satellite data layer models. 

This study implements a precise probability-based approach to replace manual delineation workflows with a scalable, data-driven machine learning solution. 

### Project Framework & Research Abstract

Differentiating sprawling infrastructure profiles from native Mediterranean topographies across multiple decades requires intense validation steps to eliminate human auditing bias. This pipeline automates the extraction process through localized spectral model definitions: 

### Balanced Sample Architecture

* **Data Dimension:** Training sets are compiled using an equal balance across two target categories (**Urban** vs. **Non-Urban** environments) to counteract network bias. 

  * **1985 Dataset:** Formatted across **2,330 independent geographic point variables**.
  * **2020 Dataset:** Formatted across **2,085 independent geographic point variables**, verified against high-resolution base reference layers.
* **Feature Scope:** Each targeted pixel coordinates into an explicit **6-dimensional space matrix**, corresponding to the specific spectral reflection channels captured by the Landsat missions. This strict matching limits model overfitting while ensuring the output remains true to physical land topography.

### Probability and Logistical Thresholding

* **Activation Mapping:** Pixel evaluations generate a continuous output index array scaled between **0 and 100**, representing the exact percentage probability of a pixel containing built-up infrastructure footprints.
* **Classification Boundary:** A sharp logical threshold limit of **50** is systematically applied across the output matrix array. Pixels registering a probability score of 50 or above are classified as urban land structures, effectively producing clean binary classification layer map assets.

### Core Technical Architecture

* **Network Blueprint:** Employs a local patternnet neural topography structured with **10 hidden processing layers** as recommended by optimal configuration standards.
* **Algorithmic Ingestion Suite:** Configures **Scaled Conjugate Gradient Backpropagation** (trainscg) as the primary training mechanism for stable weight adjustments across feature profiles.
* **Loss Function Integration:** Employs **Cross-Entropy** loss calculation during optimization epochs to minimize error values at the terminal dataset computation loop.
* **Random Data Partitioning:** Automatically splits training matrices (70% Training, 15% Validation, 15% Testing) to guarantee independent validation checks on the output layers.

