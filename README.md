# DeepVS ACM-BCB 2022

This repo contains the PyTorch implementation of DeepVS model introduced in the paper DeepVS: A Deep Learning Approach For RF-based Vital Signs Sensing.

## Repo structure

The repository structure is described below. 

```
├── README.md                                 : Description of this repository
├── models                                    : folder contains model design.
│   ├── __init__.py                           : init
│   ├── conv_attn_torch.py                    : model design of conv_attn 
├── checkpoints                               : config files and checkpoints of the pretrained models
│   ├── cnv2_attn_ts_rrhr_1                   : folder of two-stream conv2_attn1
│   ├── cnv2_attn_ts_rrhr_v_1                 : folder of two-stream conv2_attn1_v
├── LICENSE                                   : We use MIT license to manage copyright

```


## Acknowledgement

This model design was built based on the codebase [healthcare-dev](https://github.com/mit-han-lab/healthcare-dev) maintained by [Hanrui Wang](https://hanruiwang.me/) from [MIT Prof Han's Lab](https://tinyml.mit.edu/).

## Reference
If you find the repo useful, please kindly cite our paper:
```
Zongxing Xie, Hanrui Wang, Song Han, Elinor Schoenfeld, and Fan Ye. 2022.
DeepVS: A Deep Learning Approach For RF-based Vital Signs Sensing. In
13th ACM International Conference on Bioinformatics, Computational Biology
and Health Informatics (BCB ’22), August 7–10, 2022, Northbrook, IL, USA.
ACM, New York, NY, USA, 5 pages. https://doi.org/10.1145/3535508.3545554
```

Other papers related to RF-based vital signs monitoring:

- VitalHub: Robust, Non-Touch Multi-User Vital Signs Monitoring using Depth Camera-Aided UWB ([IEEE ICHI 2021](https://ieeexplore.ieee.org/abstract/document/9565710))

- Signal quality detection towards practical non-touch vital sign monitoring ([ACM-BCB 2021](https://dl.acm.org/doi/abs/10.1145/3459930.3469526))
