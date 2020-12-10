Intel® Low Precision Optimization Tool
======================================

# Introduction

Intel® Low Precision Optimization Tool is an open-source Python library that delivers a unified low-precision inference interface across multiple Intel-optimized DL frameworks on both CPUs and GPUs. It supports automatic accuracy-driven tuning strategies, along with additional objectives such as optimizing for performance, model size, and memory footprint. It also provides easy extension capability for new backends, tuning strategies, metrics, and objectives.

> **Note**
>
> GPU support is under development.

# Install from source for Windows

## Pre-Installation Requirements

The following prerequisites and requirements must be satisfied in order for the to install successfully.

Python = 3.7

[Anaconda](https://anaconda.org/)

Install packages below in anaconda:

```shell
conda create -n ilit python=3.7
activate ilit
conda install -y Cython && conda install -y numpy && conda install -y pandas && conda install -y pyyaml && conda install -c conda-forge schema && conda install -c conda-forge py-cpuinfo && conda install -y scikit-learn && conda install -y matplotlib && conda install -c conda-forge hyperopt && conda install -c conda-forge contextlib2 && conda install -c intel intel-tensorflow

pip install git+https://github.com/philferriere/cocoapi.git#egg=pycocotools^&subdirectory=PythonAPI
```

## Installation Procedure

Install ilit from source:

```shell
git clone https://github.com/intel/lp-opt-tool.git
cd lp-opt-tool
python setup.py install --user
```
## Post-Installation Tasks

After you’ve completed the installation, you want to check that everything works fine.

To ensure this, let’s check that everything works correctly.

# TroubleShooting

In this section, we help users fix common issues that may arise.

- Problem
No local packages or working download links found for pycocotools.
- Fix
  - Download [Visual C++](https://visualstudio.microsoft.com/zh-hans/thank-you-downloading-visual-studio/?sku=BuildTools&rel=15#) and then install Visual C++ in default.
  - Download cocoapi from [github](https://links.jianshu.com/go?to=https%3A%2F%2Fgithub.com%2Fphilferriere%2Fcocoapi), cd cocoapi\PythonAPI, run:
  ```shell
  # install pycocotools locally
  python setup.py build_ext --inplace

  # install pycocotools to the Python site-packages
  python setup.py build_ext install

- Problem
When importing ilit in python, this error occured in networkx package:  << OLE Object: Picture (Device Independent Bitmap) >>
- Fix
Add below cmds:
```shell
conda install network
conda install -c conda-forge hyperopt
conda install -c intel tensorflow
```




