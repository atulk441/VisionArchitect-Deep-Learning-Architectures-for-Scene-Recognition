# VisionArchitect-Deep-Learning-Architectures-for-Scene-Recognition

## Abstract
This study investigates the performance of various deep learning architectures on the Intel Image Classification Dataset, a multi-class dataset of natural and urban scene images. Our objective is to identify the most effective model for scene recognition by comparing six different approaches: a baseline StandardCNN, an AutoEncoder with a classification head, a CNN with integrated attention mechanisms, a Residual Network ResNet, Feature Pyramid Networks FPN, and several pre-trained models including CLIP and ResNet variants. We perform extensive hyperparameter tuning and evaluate each model using metrics such as accuracy, precision, recall, and F1-score. Our findings show that pretrained models outperform models trained from scratch, with CLIP-L14 achieving the highest validation accuracy of 96% in frozen configuration, followed by CLIP-B32 at 95%. Among models trained from scratch, attention mechanisms, particularly BAM achieved the best performance at 91.5%. These results underscore the importance of architectural choices and transfer learning in achieving state-of-the-art performance in image classification tasks.

This study was executed over SHARCNET's high-perfromance computing infrastructure, utilizing NVIDIA A100 and T4 GPUs. Training and hyperparameter tuning spanned multiple days across dozens of experimental configurations, encompassing both from-scratch and transfer-learning paradigms. 

## Experimental Framework
- **Compute Infrastructure:** SHARCNET HPC Cluster, NVIDIA A100 (80GB) & T4 GPUs
- **Dataset:** Intel Image Classification Dataset (25,000 images across 6 scene classes)
- **Total Experimenets:** 270+ model-hyperparameter configurations
- **Optimization Strategies:** SGD, Adam, AdamW, Dropout tuning, Batch Normalization, and architectural ablations

## Models and Architectures Compared
1. **Standard CNN (Baseline):** Depth variantions (3, 5, 7 layers) with optimizer and pooling ablations
2. **AutoEncoder with Classification Head:** Two-stage unsupervised + supervised hybrid
3. **CNN with Attention:** Evaluated CBAM, BAM, SE, Double and Triplet Attention modules
4. **Residual Networks (ResNet):** ResNet-18 to ResNet-152 with and without data augumentation
5. **Feature Pyramid Network (FPN):** Multi-scale feature fusion on top of prior architectures
6. **Pre-trained Models:**
   - OpenAI's CLIP (B/32, L/14)
   - EfficientNet-B7
   - DenseNet-201
   - Swin-V2-B
   - ConvNeXt-Large
   - ResNet Variants (18, 152)

## Highlights of Research
1) CLIP-L14 (Frozen) achieved the highest validation accuracy of 96% and 95.3% on the test set.
2) BAM-Attention CNN achieved the best performance among models trained from scratch i.e. 91.5% validation accuracy.
3) Pre-trained architectures significantly outperformed models trained from scratch, emphasizing the dominance of transfer learning in visual recognition.
4) Adam optimizer provided the most stable convergance and superior generalization across CNN variants.
5) Achieved near state-of-the-art accuracy for multi-class natural scene recognition.
