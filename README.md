# PCB-RandNet: Rethinking Random Sampling for LIDAR Semantic Segmentation in Autonomous Driving Scene


Code for paper:
> **PCB-RandNet: Rethinking Random Sampling for LIDAR Semantic Segmentation in Autonomous Driving Scene**
> <br>Huixian Cheng, Xianfeng Han, Guoqiang Xiao<br>
> Hope to be accepted by ~~AAAI 2023~~

<div align="center">
  <img src="assert/ppl.png"/>
</div><br/>

## Abstract
Fast and efficient semantic segmentation of large-scale LiDAR point clouds is a fundamental problem in autonomous driving. To achieve this goal, the existing point-based methods mainly choose to adopt Random Sampling strategy to process large-scale point clouds. However, our quantative and qualitative studies have found that Random Sampling may be less suitable for the autonomous driving scenario, since the LiDAR points follow an uneven or even long-tailed distribution across the space, which prevents the model from capturing sufficient information from points in different distance ranges and reduces the model's learning capability. To alleviate this problem, we propose a new Polar Cylinder Balanced Random Sampling method that enables the downsampled point clouds to maintain a more balanced distribution and improve the segmentation performance under different spatial distributions. In addition, a sampling consistency loss is introduced to further improve the segmentation performance and reduce the model's variance under different sampling methods. Extensive experiments confirm that our approach produces excellent performance on both SemanticKITTI and SemanticPOSS benchmarks, achieving a 2.8% and 4.0% improvement, respectively.


## Environment Setup
Install python packages
```
conda env create -f my_env.yaml
conda activate randla
```
Note: 
- Not all packages are necessary, you can refer to this environment to install.
- Since we are using operator `torch.take_along_dim` that are only available in pytorch version ≥ 1.9.0 in [here](https://github.com/huixiancheng/PCB-RandNet/blob/3d05d457761f5f096263b9658940bc7d01ce04eb/train_both_SemanticKITTI.py#L170), please make sure that the installed version meets the requirements.

Compile C++ Wrappers
```
bash compile_op.sh
```

## Prepare Data
Download SemanticKITTI from [official web](http://www.semantic-kitti.org/dataset.html). Download SemanticPOSS from [official web](http://www.poss.pku.edu.cn./download.html).
Then preprocess the data:

```
python data_prepare_semantickitti.py
python data_prepare_semanticposs.py
```
Note: 
- Please change the data path with your own path.
- Change corresponding path `self.dataset_path` in all `.py` in dataset (such as [here](https://github.com/huixiancheng/PCB-RandNet/blob/3d05d457761f5f096263b9658940bc7d01ce04eb/dataset/semkitti_trainset.py#L32)).

## Training

1. Training baseline with RS or PCB-RS

```
python train_SemanticKITTI.py <args> 

python train_SemanticPOSS.py <args>

Options:
--backbone           select the backbone to be used: choices=['randla', 'baflac', 'baaf']
--checkpoint_path    path to pretrained models(if any), otherwise train from start
--log_dir            Name of the log dir
--max_epoch          max epoch for the model to run, default 80
--batch_size         training batch size, default 6 (indicated in oroginal implementation), modify to full utilize the GPU/s
--val_batch_size     batch size for validation, default 30
--num_workers        number of workers for I/O
--sampling           select the sampling way: RS or PCB-RS. choices=['random', 'polar']
--seed               set random seed
--step               set length of dataset: 0 mean use all data || 4 mean use 1/4 dataset
--grid               resolution of polar cylinder
```

2. Training model with PCB-RS and SCL
```
python train_both_SemanticKITTI.py <args>

python train_both_SemanticPOSS.py <args>

Options:  Similar to before.
```

## Test

```
python test_SemanticKITTI.py <args>

python test_SemanticPOSS.py <args>

Options:
--infer_type         all: infer all points in specified sequence, sub: subsamples in specified sequence
--sampling           select the sampling way: RS or PCB-RS. choices=['random', 'polar']
--backbone           select the backbone to be used: choices=['randla', 'baflac', 'baaf']
--checkpoint_path    required. path to the model to test
--test_id            sequence id to test
--result_dir         result dir of predictions
--step               set length of dataset: 0 mean use all data || 4 mean use 1/4 dataset (Only use for SemanticKITTI)
--grid               resolution of polar cylinder
```
**Note:** 
- **For SemanticKITTI dataset, if your want to infer all data of 08 sequence, please make sure you have more than 32G of free RAM.**

- **Due to the use of `torch_data.IterableDataset`, the num_workers needs to be set to 0 and cannot use multi-threaded acceleration, so the test speed will be very slow. Welcome to propose a feasible accelerated PR for this. (This [reference](https://medium.com/speechmatics/how-to-build-a-streaming-dataloader-with-pytorch-a66dd891d9dd) may be useful)**

## Other Utils
1. Evaluation with Distances
```
python evaluate_DIS_SemanticKITTI.py <args>     ||      python evaluate_DIS_SemanticPOSS.py <args>
```
2. Visualization
```
python visualize_SemanticKITTI.py <args>
```
This code is not used, so it has not been tested and may have bugs. If you want to use this code to visualize SemanticKITTI and SemanticPOSS, please refer to [this repo](https://github.com/PRBonn/semantic-kitti-api) and [this repo](https://github.com/huixiancheng/CENet).

3. Others in tool.
```
caculate_time.py            Use to get Time Consumption Comparison between RS, PCB-RS, and FPS.
draw.py / draw_poss.py      Use to draw performance plot at different distances.
draw_vis_compare.py         Use to generate qualitative visualization and quantitative statistical analysis images between RS and PCB-RS similar to the paper.
eval_KITTI_gap.py           Use to calculate the difference in performance of the model under different sampling methods.
main_poss.py                Use to count distance distribution
```

## Others to Note and Clarify

- All our experiments were trained on a single NVIDIA RTX 3090 GPU using mixed precision.
- We set `torch.backends.cudnn.enabled = False` and use `with torch.cuda.amp.autocast():` to reduce training time and save GPU memory. However, it is not certain whether it will have an impact on performance.
- In `train_SemanticKITTI.py`, We use `num_classes = 20` instead of `num_classes = 19`. The reason is that I think the ops used to calculate the loss in the original code is very complex, and its main point is to ignore the loss when `label == 0`. Therefore, I just changed the class number to 20 and set the weight of `label == 0` to 0 to achieve it.
- 

## Acknowledgement

- Original Tensorflow implementation [link](https://github.com/QingyongHu/RandLA-Net)
- Our network & config codes are modified from [RandLA-Net PyTorch](https://github.com/qiqihaer/RandLA-Net-pytorch)
- Our evaluation & visualization codes are modified from [SemanticKITTI API](https://github.com/PRBonn/semantic-kitti-api)
