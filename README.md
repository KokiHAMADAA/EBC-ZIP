# EBC-ZIP

The official implementation of the paper [*EBC-ZIP: Improving Blockwise Crowd Counting with Zero-Inflated Poisson Regression*](https://arxiv.org/pdf/2506.19955).

## Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

## Step 2: Download Processed Datasets


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