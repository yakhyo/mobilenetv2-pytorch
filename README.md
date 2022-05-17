## Implementation of [MobileNetV2](https://arxiv.org/abs/1801.04381) in PyTorch

**Arxiv**: https://arxiv.org/abs/1801.04381

After 450 epochs MobileNetV3L reaches **Acc@1**: 70.1926 **Acc@5**: 89.6662

### Updates

* 2022.05.18:

    - Weights are uploaded to the `weights` folder. `last.ckpt` is checkpoint (56.7MB) (includes model, model_ema, optimizer, ...) and last.pth is model with

      Exponential Moving Average (7.3MB) and converted to `half()` tensor.

### Dataset

Specify the IMAGENET data folder in the `main.py` file.

``` python
parser.add_argument("--data-path", default="../../Projects/Datasets/IMAGENET/", type=str, help="dataset path")
```

IMAGENET folder structure:

```
├── IMAGENET 
    ├── train
         ├── [class_id1]/xxx.{jpg,png,jpeg}
         ├── [class_id2]/xxy.{jpg,png,jpeg}
         ├── [class_id3]/xxz.{jpg,png,jpeg}
          ....
    ├── val
         ├── [class_id1]/xxx1.{jpg,png,jpeg}
         ├── [class_id2]/xxy2.{jpg,png,jpeg}
         ├── [class_id3]/xxz3.{jpg,png,jpeg}
```

### Train

Run `main.sh` (for DDP) file by running the following command:

```
bash main.sh
```

`main.sh`:

```
torchrun --nproc_per_node=@num_gpu main.py --epochs 300  --batch-size 512 --lr 0.064  --lr-step-size 2 --lr-gamma 0.973 --random-erase 0.2
```

Run `main.py` for `DataParallel` training.

The training config taken
from [official torchvision models' training config](https://github.com/pytorch/vision/tree/970ba3555794d163daca0ab95240d21e3035c304/references/classification)
.

### Evaluation
```commandline
python main.py --test
```
