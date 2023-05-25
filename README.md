**POET**: End-to-End Multi-Instance Pose Estimation with Transformers
========
This repository contains the official implementation of **POET** (**PO**se **E**stimation **T**ransformer) and is built on top of [DETR](https://github.com/facebookresearch/detr).
We replace the full complex hand-crafted pose estimation pipeline with a Transformer, and outperform Associative Embedding with a ResNet-50, obtaining **54 mAP** on COCO.

![POET](.github/POET.png)


For details see [End-to-End Trainable Multi-Instance Pose Estimation with Transformers](https://arxiv.org/abs/2103.12115) by Lucas Stoffl, Maxime Vidal and Alexander Mathis.



# Usage
To train a POET model on a single node with 2 GPUs for 250 epochs run:
```
python -m torch.distributed.launch --nproc_per_node=2 --use_env  main.py --batch_size=6 --num_workers=16 --epochs=250 --coco_path=data/COCO --set_cost_kpts=4 --set_cost_ctrs=0.5 --set_cost_deltas=0.5 --set_cost_kpts_class=0.2 --kpts_loss_coef=4 --ctrs_loss_coef=0.5 --deltas_loss_coef=0.5 --kpts_class_loss_coef=0.2 --num_queries=50 --output_dir=experiments/ --lr=5e-5 --lr_backbone=5e-6
```

Download of a trained model:
[POET-R50](https://zenodo.org/record/7972042)

# Notebooks

We furthermore provide a demo notebook to help you get a grasp on POET:
* [POET's demo notebook](notebooks/poet_demo.ipynb): Shows how to load a pre-trained model, generate predictions and visualize the attention of the model.


# License
POET is released under the Apache 2.0 license. Please see the [LICENSE](LICENSE) file for more information.

