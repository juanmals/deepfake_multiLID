# Detection Synthetic Diffusion Generated Images Using multi Local Intrinsic Dimensionality

<b>Peter Lorenz, Ricard Durall, and Janis Keuper</b>

[[Paper](https://arxiv.org/pdf/2307.02347.pdf)]

Please cite:

```
@article{
      lorenz2023detecting,
      title={Detecting Images Generated by Deep Diffusion Models using their Local Intrinsic Dimensionality}, 
      author={Peter Lorenz and Ricard Durall and Janis Keuper},
      journal = {ICCV Workshop and Challenge on DeepFake Analysis and Detection},
      year={2023},
}
```


## Abstract
Diffusion models have recently been successfully applied for the visual synthesis of remarkably realistic images. However, this raises significant concerns about their potential misuse for malicious purposes. In this paper, we propose a solution using the lightweight multi Local Intrinsic Dimensionality (multiLID) method, originally developed for detecting adversarial examples, to automatically detect synthetic images and identify the corresponding generator networks.
Unlike many existing detection approaches that may only work effectively for GAN-generated images, our proposed method achieves close-to-perfect detection results in various realistic use cases. Through extensive experiments on known and newly created datasets, we demonstrate the superiority of the multiLID approach in diffusion detection and model identification. Additionally, while recent publications primarily focus on the "LSUN-Bedroom" dataset for evaluating the detection of generated images, we establish a comprehensive benchmark for the detection of diffusion-generated images. This benchmark includes samples from several diffusion models with different image sizes.

<!-- <p align="center">
<img src="figs/teaser.png" width=60%>
</p> -->

## multiLID
<p align="center" width="100%">
  <img src="./assets/teaser.png" alt="teaser multiLID" />
</p>

The underlying concept of the proposed method is to distinguish models by differences in the density of their internal feature distributions. LID estimates densities in the feature spaces of pre-trained CNNs, by computing fractions over the number of samples in given volumes: ∣volume I∣/∣volume II∣ < 1. The example above shows how this density measure indicates if the selected sample belong (left) or does not belong (right) to a reference distribution.

<p align="center" width="100%">
  <img src="./assets/compresults.png" alt="results" />
</p>

Limitation of the identification. Left: As described in section 4.1, our experiment is based on the ArtiFact and consists of 8 clean datasets, 6 GAN, and 6 DM-generated images. Center: Identification results on the dataset of LSUN-Bedroom. Right: Transferability results on the dataset of LSUN-Bedroom. The transferability is low, while the identification between clean and synthetic images is accurate. 


## TODO
- [ ] Release code.

## Acknowledgments
Our code is developed based on [multiLID](https://arxiv.org/pdf/2212.06776.pdf), [DDPM](https://arxiv.org/abs/2006.11239) and [huggingface](https://huggingface.co/). 
Thanks for their sharing codes and models.
