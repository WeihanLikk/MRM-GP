> [!IMPORTANT]
> **A newer and more efficient implementation is now available at [MBRILA](https://github.com/BRAINML-GT/MBRILA).**
> MBRILA is a GPU-native PyTorch library for multi-region neural latent variable models. It builds on our follow-up work ([Li et al., ICML 2025](https://proceedings.mlr.press/v267/li25ck.html)), which supports time-varying inter-region delays with O(log T) parallel inference, and it also includes DLAG, mDLAG, GPFA and more under a unified interface. We recommend using MBRILA for new projects; this repository is kept for reproducing the ICML 2024 results.

## [ICML 2024] Multi-Region Markovian Gaussian Process: An Efficient Method to Discover Directional Communications Across Multiple Brain Regions 

<div align='center' >Weihan Li, Chengrui Li, Yule Wang, and Anqi Wu</div> 

[[arXiv]](https://arxiv.org/abs/2402.02686)

## 1 Installation
From the current directory, run
```
pip install -r requirements.txt
```

Our code is highly dependent on the [SSM](https://github.com/lindermanlab/ssm) package. To install SSM, run
```
git clone https://github.com/lindermanlab/ssm
cd ssm
pip install numpy cython
pip install -e .
```

## 2 Tutorial

`demo.ipynb` and `demo_r2.ipynb` are two step-by-step tutorials that learn an MRMGP from a demo dataset and evaluate via log-likelihood on held-out neurons and R square on predictions.
