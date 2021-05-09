# Development of a Deep Learning Model for Early Alzheimer’s Disease Detection from Structural MRIs and External Validation on an Independent Cohort 

## Introduction
Early diagnosis of Alzheimer’s disease plays a pivotal role in patient care and clinical trials. In this study, we have developed a new approach based on 3D deep convolutional neural networks to accurately differentiate mild Alzheimer’s disease dementia from mild cognitive impairment and cognitively normal individuals using structural MRIs. For comparison, we have built a reference model based on the volumes and thickness of previously reported brain regions that are known to be implicated in disease progression. We validate both models on an internal held-out cohort from The Alzheimer's Disease Neuroimaging Initiative (ADNI) and on an external independent cohort from The National Alzheimer's Coordinating Center (NACC). The deep-learning model is more accurate and significantly faster than the volume/thickness model. The model can also be used to forecast progression: subjects with mild cognitive impairment misclassified as having mild Alzheimer’s disease dementia by the model were faster to progress to dementia over time. An analysis of the features learned by the proposed model shows that it relies on a wide range of regions associated with Alzheimer's disease. These findings suggest that deep neural networks can automatically learn to identify imaging biomarkers that are predictive of Alzheimer's disease, and leverage them to achieve accurate early detection of the disease.

## Model Training


### Prerequisites

- Python 3.6
- PyTorch 0.4
- torchvision
- progress
- matplotlib
- numpy
- visdom


### Data

The IDs of patients and scans used in our study and training, validation, test as well as Freesurfer segmentation volume and thickness features for ADNI and NACC are available under folder

```
Data
```


### Train

train by running command:

```
python src/train.py
```
You can create your own config files and add a --config flag to indicate the name of your config files.


### Model Evaluation
The trained deep learning model can be found [here](https://drive.google.com/file/d/1zU21Kin9kXg_qmj7w_u5dGOjXf1D5fa7/view?usp=sharing).



## Model Overview

<p float="left" align="center">
<img src="overview.png" width="800" /> 
<figcaption align="center">
Figure: Overview of the deep learning framework and performance for Alzheimer’s automatic diagnosis. (a) Deep learning framework used for automatic diagnosis. (b) Receiver operating characteristic (ROC) curves for classification of cognitively normal (CN), mild cognitive impairment (MCI) and Alzheimer’s disease (AD), computed on the ADNI held-out test set. (c) ROC curves for classification of cognitively normal (CN), mild cognitive impairment (MCI) and Alzheimer’s disease (AD) on the NACC test set. (d) Visualization using t-SNE projections of the features computed by the proposed deep-learning model. Each point represents a scan. Green, blue, red colors indicate predicted cognitive groups. CN and AD scans are clearly clustered. (e) Visualization using t-SNE projections of the 138 volumes and thickness in the ROI-volume/thickness model. Compared to (d) the separation between CN and AD scans is less marked. The t-SNE approach is described in details in the methods section.

