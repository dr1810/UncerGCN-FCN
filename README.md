# UncerGCN-FCN: An Uncertainty-Aware Ensemble Model Combining Graph Convolutional and Fully Connected Networks for Predicting pIC50 Values

## Overview

This project focuses on predicting molecular properties, specifically the **pIC50** values, which are a critical measure of a molecule's potency as a drug candidate. The project utilizes two primary deep learning architectures:
- **Graph Convolutional Networks (GCNs)** for graph-structured molecular data.
- **Fully Connected Neural Networks (FCNNs)** for feature-based molecular data.

To improve predictive accuracy, an ensemble model combining both GCN and FCNN predictions, named **UncerGCN-FCN**, is developed. The ensemble model leverages the strengths of both architectures to make a more robust and accurate prediction.

## Table of Contents
1. [Introduction to Molecular Property Prediction Using GCN and FCNN](#introduction)
2. [Graph Convolutional Network (GCN) for Molecular Property Prediction](#gcn)
3. [Fully Connected Neural Network (FCNN) for Molecular Property Prediction](#fcnn)
4. [UncerGCN-FCN for Enhanced Molecular Property Prediction](#ensemble)
5. [Conclusion](#conclusion)
6. [References](#references)
7. [Appendix](#appendix)

## 1. Introduction to Molecular Property Prediction Using GCN and FCNN

The primary goal of this project is to predict **pIC50 values**, which serve as a measure of a molecule’s efficacy as a drug. These values are critical in the field of drug discovery, helping researchers identify and optimize potential drug candidates.

### Model Architectures:
- **GCNs**: Used to process graph-based representations of molecules, where atoms are represented as nodes and bonds as edges. The model captures the complex relationships between atoms in the molecule.
- **FCNNs**: Used for feature-based representations of molecules, where molecular properties are encoded into numerical features like molecular fingerprints.

## 2. Graph Convolutional Network (GCN) for Molecular Property Prediction

The GCN is designed to handle graph-structured molecular data. The input is a molecular graph with nodes representing atoms and edges representing bonds.

### Model Architecture:
- **Input Layer**: A tensor representing the molecular graph.
- **Graph Convolution Layers**: Two layers, each refining the atom representations.
- **Dropout Layers**: Used for regularization.
- **Global Average Pooling**: Reduces node features into a single tensor representation.
- **Fully Connected Layer**: Outputs the predicted pIC50 value.

### Training Results:
- **Training Loss**: 1.3568
- **Validation Loss**: 1.4993
- **Training MAE**: 1.0436
- **Validation MAE**: 1.0865

## 3. Fully Connected Neural Network (FCNN) for Molecular Property Prediction

The FCNN model works with feature-based molecular data, where molecular descriptors are used as input.

### Model Architecture:
- **Input Layer**: A vector of molecular features.
- **Dense Layers**: Two hidden layers with ReLU activation.
- **Dropout Layers**: Used for regularization.
- **Output Layer**: A scalar output for the predicted pIC50 value.

### Training Results:
- **Training Loss**: 1.1205
- **Validation Loss**: 1.2256
- **Training MAE**: 0.9775
- **Validation MAE**: 1.0223

## 4. UncerGCN-FCN for Enhanced Molecular Property Prediction

The **UncerGCN-FCN** model is an ensemble of the GCN and FCNN models, where the predictions from both models are combined to provide a more accurate final output.

### Model Architecture:
- **GCN Sub-model**: Provides predictions based on graph data.
- **FCNN Sub-model**: Provides predictions based on feature-based data.
- The predictions are combined and passed through a final prediction layer.

### Training Results:
- **Training Loss**: 1.0136
- **Validation Loss**: 1.1453
- **Training MAE**: 0.9312
- **Validation MAE**: 0.9732

## 5. Conclusion

The ensemble model, UncerGCN-FCN, outperforms both the individual GCN and FCNN models, showcasing the effectiveness of ensemble learning. By combining the strengths of graph-based and feature-based data processing, the ensemble model provides more robust and accurate predictions.

### Key Takeaways:
- **GCN**: Best for capturing molecular interactions in graph-structured data.
- **FCNN**: Effective for feature-based molecular representations.
- **Ensemble Model**: Achieves the best performance by combining both GCN and FCNN predictions.

## 6. References

1. El-Behery, H., Attia, A. F., El-Fishawy, N., & Torkey, H. (2021). Efficient machine learning model for predicting drug-target interactions with case study for Covid-19. *Computational Biology and Chemistry*, 93, 107536.
2. Tayara, H., Abdelbaky, I., & To Chong, K. (2021). Recent omics-based computational methods for COVID-19 drug discovery and repurposing. *Briefings in Bioinformatics*, 22(6), bbab339.
3. Zhang, Y., Ye, T., Xi, H., Juhas, M., & Li, J. (2021). Deep learning driven drug discovery: tackling severe acute respiratory syndrome coronavirus 2. *Frontiers in Microbiology*, 12, 739684.
4. Zeng, X., Song, X., Ma, T., Pan, X., Zhou, Y., Hou, Y., ... & Cheng, F. (2020). Repurpose open data to discover therapeutics for COVID-19 using deep learning. *Journal of Proteome Research*, 19(11), 4624-4636.
5. Pham, T. H., Qiu, Y., Zeng, J., Xie, L., & Zhang, P. (2021). A deep learning framework for high-throughput mechanism-driven phenotype compound screening and its application to COVID-19 drug repurposing. *Nature Machine Intelligence*, 3(3), 247-257.
6. Patel, L., Shukla, T., Huang, X., Ussery, D. W., & Wang, S. (2020). Machine learning methods in drug discovery. *Molecules*, 25(22), 5277.

## Installation

To run this project, you need to install the following dependencies:

```bash
pip install tensorflow numpy pandas scikit-learn matplotlib networkx
