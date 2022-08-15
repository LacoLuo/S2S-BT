# Sequence-to-Sequence Learning for Beam Tracking
- The official implementation of the VTC2021-Fall paper [Machine Learning Based mmWave Orchestration for Edge Gaming QoE Enhancement](https://ieeexplore.ieee.org/document/9625307).
- The repository uses the [ViWi](https://viwi-dataset.net) dataset.
## Requirements
- Python 3.6
- Pytorch 1.4
## Get Started
### Dataset
1. Choose a scenario of the ViWi dataset.
2. Generate the raw dataset by following the official instruction.
3. Create a codebook.
4. Use the raw dataset and the codebook to find the optimal beam index of each data point.
```
python preprocess.py \
  --image_dir scenario/rgb/ \
  --codebook data_generation_package/codebook \
  --wireless data_generation_package/data/raw_data 
```
5. Generate the beam tracking dataset in csv format. Each data sample contains 13 consecutive beam indices.
```
python generate_dataset.py \
  --beam_dir beam_dir/ 
```
Example:
```
Beam_1,Beam_2,Beam_3,Beam_4,Beam_5,Beam_6,Beam_7,Beam_8,Beam_9,Beam_10,Beam_11,Beam_12,Beam_13
114,114,113,113,113,113,113,112,112,112,112,112,112
```
7. Split the dataset into training, validation, and testing sets.

### Training
```
python train.py \
  --trn_data_path data/train_set.csv \
  --val_data_path data/val_set.csv \
  --store_model_path ckpt/
```
### Evaluation
```
python inference.py \ 
  --test_data_path data/test_set.csv \
  --load_model_path ckpt/best
```
## Citation
If you find our code helpful for your research, please consider citing the following paper:
```
@inproceedings{luo2021machine,
  title={Machine Learning Based mmWave Orchestration for Edge Gaming QoE Enhancement},
  author={Luo, Hao and Wei, Hung-Yu},
  booktitle={2021 IEEE 94th Vehicular Technology Conference (VTC2021-Fall)},
  pages={1--6},
  year={2021},
  organization={IEEE}
}
```
## Acknowlegements
- [https://github.com/malrabeiah/VABT](https://github.com/malrabeiah/VABT)
- [https://github.com/malrabeiah/CameraPredictsBeams](https://github.com/malrabeiah/CameraPredictsBeams)
