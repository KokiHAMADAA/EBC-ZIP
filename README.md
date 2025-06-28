# EBC-ZIP

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/ebc-zip-improving-blockwise-crowd-counting/crowd-counting-on-shanghaitech-a)](https://paperswithcode.com/sota/crowd-counting-on-shanghaitech-a?p=ebc-zip-improving-blockwise-crowd-counting)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/ebc-zip-improving-blockwise-crowd-counting/crowd-counting-on-shanghaitech-b)](https://paperswithcode.com/sota/crowd-counting-on-shanghaitech-b?p=ebc-zip-improving-blockwise-crowd-counting)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/ebc-zip-improving-blockwise-crowd-counting/crowd-counting-on-ucf-qnrf)](https://paperswithcode.com/sota/crowd-counting-on-ucf-qnrf?p=ebc-zip-improving-blockwise-crowd-counting)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/ebc-zip-improving-blockwise-crowd-counting/crowd-counting-on-nwpu-crowd-val)](https://paperswithcode.com/sota/crowd-counting-on-nwpu-crowd-val?p=ebc-zip-improving-blockwise-crowd-counting)

The official implementation of the paper [*EBC-ZIP: Improving Blockwise Crowd Counting with Zero-Inflated Poisson Regression*](https://arxiv.org/pdf/2506.19955).

## Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

## Step 2: Download Processed Datasets

- **ShanghaiTech A**: [sha.zip](https://github.com/Yiming-M/EBC-ZIP/releases/download/dataset/sha.zip)
- **ShanghaiTech B**: [shb.zip](https://github.com/Yiming-M/EBC-ZIP/releases/download/dataset/shb.zip)
- **UCF-QNRF**: [qnrf.zip](https://github.com/Yiming-M/EBC-ZIP/releases/download/dataset/qnrf.zip), [qnrf.z01](https://github.com/Yiming-M/EBC-ZIP/releases/download/dataset/qnrf.z01)
- **NWPU-Crowd**: [nwpu.zip](https://github.com/Yiming-M/EBC-ZIP/releases/download/dataset/nwpu.zip), [nwpu.z01](https://github.com/Yiming-M/EBC-ZIP/releases/download/dataset/nwpu.z01), [nwpu.z02](https://github.com/Yiming-M/EBC-ZIP/releases/download/dataset/nwpu.z02), [nwpu.z03](https://github.com/Yiming-M/EBC-ZIP/releases/download/dataset/nwpu.z03), [nwpu.z04](https://github.com/Yiming-M/EBC-ZIP/releases/download/dataset/nwpu.z04), [nwpu.z05](https://github.com/Yiming-M/EBC-ZIP/releases/download/dataset/nwpu.z05), [nwpu.z06](https://github.com/Yiming-M/EBC-ZIP/releases/download/dataset/nwpu.z06), [nwpu.z07](https://github.com/Yiming-M/EBC-ZIP/releases/download/dataset/nwpu.z07), [nwpu.z08](https://github.com/Yiming-M/EBC-ZIP/releases/download/dataset/nwpu.z08)

To unzip splitted `.zip` files, 7-Zip is recommended. You can use the following command to install 7-Zip and unzip the dataset:

```bash
sudo apt update
sudo apt install p7zip-full

7z x dataset.zip
```

## Step 3: Run Training

Add the training code to `run.sh` and execute it:

```bash
sh run.sh
```

If you want to use the zero-inflated loss, set either `--reg_loss` or `--aux_loss` to `zipnll`. For example, you can set `--reg_loss zipnll` to use the zero-inflated loss for regression. 

You can use an auxillary loss to improve the performance. For example, you might want to use the pre-defined multi-scale MAE loss by setting `--aux_loss msmae` and `--scales 1 2 4`.

The DMCount loss can also be used together with the zero-inflated loss. For example, you can set `--reg_loss zipnll --aux_loss dmcount` to use both losses.


## Step 4: Test the Model

Use `test.py` or `test.sh` to test the model. You can specify the dataset, weight path, input size, and other parameters.

To generate the predicted counts on NWPU-Crowd Test, you need to use `test_nwpu.py` instead.

To visualize the results, use the `notebooks/model.ipynb` notebook.

Trained weights are also provided:
- [**ShanghaiTech A**](https://github.com/Yiming-M/EBC-ZIP/releases/tag/weights_sha)
- [**ShanghaiTech B**](https://github.com/Yiming-M/EBC-ZIP/releases/tag/weights_shb)
- [**UCF-QNRF**](https://github.com/Yiming-M/EBC-ZIP/releases/tag/weights_qnrf)
- [**NWPU-Crowd**](https://github.com/Yiming-M/EBC-ZIP/releases/tag/weights_nwpu)

Make sure to use the processed datasets and the exact commands pre-defined in `test.sh` to reproduce the same results.
