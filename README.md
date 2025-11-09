# Efficient transformer with compressed-attention for stereo image super-resolution

## Dependencies
- Python 3.9
- PyTorch 1.10.0

```
cd code
pip install -r requirements.txt
python setup.py develop
```
## Datasets
- LCATSSR

|  Training Set   | Testing Set   |
|  ----  | ----  |
|  Flickr1024 + Middlebury | KITTI2012 + KITTI2015 + Middlebury + Flickr1024  |

- LCATSR

|  Training Set   | Testing Set   |
|  ----  | ----  |
|  DIV2K | Set5 + Set14 + BSD100 + Urban100 + Manga109  |

Refer to the datasets folder for the complete data. 

## Implementation of LCATSSR
### Train

```shell
#scale factor 2
python -m torch.distributed.launch --nproc_per_node=4 --master_port=4321 basicsr/train.py -opt options/train/LCATSSR/LCATSSR_x2.yml --launcher pytorch
#scale factor 4
python -m torch.distributed.launch --nproc_per_node=4 --master_port=4321 basicsr/train.py -opt options/train/LCATSSR/LCATSSR_x4.yml --launcher pytorch
```
### Test
```shell
#scale factor 2
python -m torch.distributed.launch --nproc_per_node=1 --master_port=4321 basicsr/test.py -opt options/test/LCATSSR/LCATSSR_x2.yml --launcher pytorch
#scale factor 4
python -m torch.distributed.launch --nproc_per_node=1 --master_port=4321 basicsr/test.py -opt options/test/LCATSSR/LCATSSR_x4.yml --launcher pytorch
```

## Implementation of LCATSR
### Train

```shell
#scale factor 2
python -m torch.distributed.launch --nproc_per_node=2 --master_port=4321 basicsr/train.py -opt options/train/LCATSR/LCATSR_x2.yml --launcher pytorch
#scale factor 3
python -m torch.distributed.launch --nproc_per_node=2 --master_port=4321 basicsr/train.py -opt options/train/LCATSR/LCATSR_x3.yml --launcher pytorch
#scale factor 4
python -m torch.distributed.launch --nproc_per_node=2 --master_port=4321 basicsr/train.py -opt options/train/LCATSR/LCATSR_x4.yml --launcher pytorch
```
### Test
```shell
#scale factor 2
python test_SISR.py --scale 2 --model_path './experiments/pretrained_models/LCATSR_x2.pth'
#scale factor 3
python test_SISR.py --scale 3 --model_path './experiments/pretrained_models/LCATSR_x3.pth'    
#scale factor 4
python test_SISR.py --scale 4 --model_path './experiments/pretrained_models/LCATSR_x4.pth'  
```

## Citation
```
@article{song2025etcassr,
  title = {Efficient transformer with compressed-attention for stereo image super-resolution},
  author={Song, Jianwen and Sowmya, Arcot and Zhang, Weichuan and Sun, Changming},
  journal = {Knowledge-Based Systems},
  year = {2025}
```

