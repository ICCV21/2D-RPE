# Rethinking Relative Position Encoding for Vision Transformer

**NOTICE:**

This is the release code for ICCV submission: **Rethinking Relative Position Encoding for Vision Transformer**.

# Model Zoo

We equip DeiT models with contextual product shared-head RPE with 50 buckets, and report their accuracy on ImageNet-1K Validation set.

Model | RPE-Q | RPE-K | RPE-V | #Params(M) | MACs(M) | Top-1 Acc.(%) | Top-5 Acc.(%) | Link | Log
----- | ----- | ----- | ----- | ---------- | ------- | ------------- | ------------- | ---- | ---
tiny | | ✔ | | 5.76 | 1284 | 73.7 | 92.0 | [link](https://github.com/ICCV21/2D-RPE/releases/download/1.0/deit_tiny_patch16_224_ctx_product_50_shared_k.pth) | [log](./logs/deit_tiny_patch16_224_ctx_product_50_shared_k/log.txt), [detail](./logs/deit_tiny_patch16_224_ctx_product_50_shared_k/detail.log)
small | | ✔ | | 22.09 | 4659 | 80.9 | 95.4 | [link](https://github.com/ICCV21/2D-RPE/releases/download/1.0/deit_small_patch16_224_ctx_product_50_shared_k.pth) | [log](./logs/deit_small_patch16_224_ctx_product_50_shared_k/log.txt), [detail](./logs/deit_small_patch16_224_ctx_product_50_shared_k/detail.log)
small | ✔ | ✔ | | 22.13 | 4706 | 81.0 | 95.5 | [link](https://github.com/ICCV21/2D-RPE/releases/download/1.0/deit_small_patch16_224_ctx_product_50_shared_qk.pth) | [log](./logs/deit_small_patch16_224_ctx_product_50_shared_qk/log.txt), [detail](./logs/deit_small_patch16_224_ctx_product_50_shared_qk/detail.log)
small | ✔ | ✔ | ✔ | 22.17 | 4885 | 81.2 | 95.5 | [link](https://github.com/ICCV21/2D-RPE/releases/download/1.0/deit_small_patch16_224_ctx_product_50_shared_qkv.pth) | [log](./logs/deit_small_patch16_224_ctx_product_50_shared_qkv/log.txt), [detail](./logs/deit_small_patch16_224_ctx_product_50_shared_qkv/detail.log)
base | | ✔ | | 86.61 | 17684 | 82.3 | 95.9 | [link](https://github.com/ICCV21/2D-RPE/releases/download/1.0/deit_base_patch16_224_ctx_product_50_shared_k.pth) | [log](./logs/deit_base_patch16_224_ctx_product_50_shared_k/log.txt), [detail](./logs/deit_base_patch16_224_ctx_product_50_shared_k/detail.log)

# Usage

## Setup
Install 3rd-packages from [requirements.txt](./requirements.txt). Notice that the version of timm should be **0.3.2**, and the version of Pytorch should be equal or higher than 1.7.0.

```bash
pip install -r ./requirements.txt
```

## Data Preparation

You can download the ImageNet-1K dataset from [`http://www.image-net.org/`](http://www.image-net.org/).

The train set and validation set should be saved as the `*.tar` archives:

```
ImageNet/
├── train.tar
└── val.tar
```

Our code also supports storing images as individual files as follow:
```
ImageNet/
├── train
│   ├── n01440764
│   │   ├── n01440764_10026.JPEG
│   │   ├── n01440764_10027.JPEG
...
├── val
│   ├── n01440764
│   │   ├── ILSVRC2012_val_00000293.JPEG
```

## Training
We define the models with 2D-RPE in [`rpe_models.py`](./rpe_models.py).

For example, we train DeiT-S with contextual product relative position encoding on keys with 50 buckets, the model's name is `deit_small_patch16_224_ctx_product_50_shared_k`.

Run the following command:
```bash
python -m torch.distributed.launch --nproc_per_node=8 --use_env main.py --model deit_small_patch16_224_ctx_product_50_shared_k --batch-size 128 --data-path ./ImageNet/ --output_dir ./outputs/ --load-tar
```

You can remove the flag `--load-tar` if storing images as individual files : )

## Evaluation
The step is similar to training. Add the checkpoint path and the flag `--eval`.
```bash
python -m torch.distributed.launch --nproc_per_node=8 --use_env main.py --model deit_small_patch16_224_ctx_product_50_shared_k --batch-size 128 --data-path ./ImageNet/ --output_dir ./outputs/ --load-tar --eval --resume deit_small_patch16_224_ctx_product_50_shared_k.pth
```

## Code Structure

Our code is based on [DeiT](https://github.com/facebookresearch/deit) and [pytorch-image-models](https://github.com/rwightman/pytorch-image-models). Thank you!

File | Description
-----|------------
[`rpe_2d.py`](./rpe_2d.py) | The implementation of 2D relative position encoding
[`rpe_models.py`](./rpe_models.py) | The implementation of models with 2D-RPE
[`rpe_vision_transformer.py`](./rpe_vision_transformer.py) | We equip 2D-RPE on `Attention`, `Block`, and `VisionTransformer` modules

# License
