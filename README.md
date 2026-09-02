# A Dual Color Space Framework with Learnable Cross-Color Fusion for Illumination-Robust Multi-View Geo-Localization



## Abstract
Appearance domain shifts induced by illumination variations, weather degradation, and sensor discrepancies pose fundamental challenges to robust multi-view geo-localization. Existing approaches focus predominantly on spatial geometric alignment and feature aggregation within RGB, while information aliasing caused by inherent luminance-chrominance coupling remains largely unaddressed, restricting generalization under dynamic and complex imaging conditions. Motivated by color constancy and signal decorrelation, a dual color space fusion (DCSF) framework with learnable cross-color fusion is proposed for illumination-robust multi-view geo-localization. Complementary RGB and YCbCr features are jointly exploited, with RGB preserving detailed geometric structure and YCbCr improving appearance adaptability through luminance-chrominance decoupling. A learnable cross-color fusion module adaptively integrates features via softmax-normalized trainable weights, and an inter-color consistency loss enforces cross-space feature alignment, strengthening illumination invariance. Furthermore, illumination change Module expands the training illumination distribution, synergizing with the dual-branch architecture. Extensive evaluations on University-1652 and SUES-200 demonstrate competitive localization accuracy under conventional illumination. More importantly, exceptional generalization and robustness are exhibited under low illumination, where severe degradation is observed for all baselines. At an extreme brightness coefficient of 10\%, accuracy drops severely for all baselines, whereas the proposed DCSF maintains an AP of 93.86\%, confirming robust generalization under low illumination. https://github.com/HAORANJY/DCSF-main.


## Requisites
- Python >= 3.8
- GPU Memory >= 6G
- Pytorch 1.13.0+cu116
- Torchvision 0.14.0+cu116


