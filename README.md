# ustcthesis code

This repo holds the source code of my master's thesis.

You can check the specific branch to find the code of each chapter. **The full text of the thesis will be released soon.**

My research topic is object detection. In the thesis, I improve the object detector from three perspectives:

1. Propose a loss function based on IoU, called LIoU. LIoU combines L1 loss and IoU. Bounding box regression simulation experiment like in [DIoU](https://arxiv.org/abs/1911.08287) shows that LIoU has a faster convergence speed, which can further improve the locating accuracy of the object detector. Related code branch is prefixed with `chap3`.
2. Design a neural network longitudinal cross-layer feature fusion algorithm called PaFPN. PaFPN retains all the advantages of [FPN](https://arxiv.org/abs/1612.03144), [PANet](https://arxiv.org/abs/1803.01534), [BiFPN](https://arxiv.org/abs/1911.09070) and [NAS-FPN](https://arxiv.org/abs/1904.07392). The experimental results on EfficientDet show that PaFPN has lower algorithm complexity and better performance. Related code branch is prefixed with `chap4`.
3. Improve the general form of the loss function based on IoU, making it has the characteristic of hard example mining. All loss functions based on IoU can benefit from the improved general form. The experimental results show that the convergence of LIoU modified based on the form is more stable, and the ability of hard example mining is better than the existing algorithms dedicated to hard example mining. Related code branch is prefixed with `chap5`.

Finally, based on the above work, a object detector called CompleteDet is designed. In order to diagnose the error of CompleteDet, the code of cocoapi is expanded to support the `analyze()` interface. Related code branch is prefixed with `chap6`.

## Thanks

Some of the code are based on other repos, thanks for their work:

1. [mmdetection](https://github.com/open-mmlab/mmdetection): a framework containing many object detection algorithms, which can be used out of the box.
2. [efficientdet-pytorch](https://github.com/rwightman/efficientdet-pytorch): EfficientDet implemented using pytorch.
3. [cocoapi](https://github.com/cocodataset/cocoapi): toolbox to calculate mAP of COCO dataset.
4. [flops-counter.pytorch](https://github.com/sovrasov/flops-counter.pytorch): script to calculate the complexity of pytorch model.

## Citation

If you use the code in the repo, please cite the thesis:

```BibTeX
@mastersthesis{zhai2021ustcthesis,
  author  = {Zhai Hongyu},
  title   = {Research on Accurate Locating and Balanced Learning Algorithms in Object Detection},
  school  = {University of Science and Technology of China},
  address = {Hefei, China},
  year    = {2021}
}
```
