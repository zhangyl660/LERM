# Rethinking Guidance Information to Utilize Unlabeled Samples: A Label-Encoding Perspective (ICML'24)

Official Implementation of Rethinking Guidance Information to Utilize Unlabeled Samples: A Label-Encoding Perspective.

Yulong Zhang, Yuan Yao, Shuhao Chen, Pengrong Jin, Yu Zhang, Jian Jin, Jiangang Lu.


## Abatract
Empirical Risk Minimization (ERM) is fragile in scenarios with insufficient labeled samples. A vanilla extension of ERM to unlabeled samples is Entropy Minimization (EntMin), which employs the soft-labels of unlabeled samples to guide their learning. However, EntMin emphasizes prediction discriminability while neglecting prediction diversity. To alleviate this issue, in this paper, we rethink the guidance information to utilize unlabeled samples. By analyzing the learning objective of ERM, we find that the guidance information for the labeled samples in a specific category is the corresponding *label encoding*. Inspired by this finding, we propose a Label-Encoding Risk Minimization (LERM). It first estimates the label encodings through prediction means of unlabeled samples and then aligns them with their corresponding ground-truth label encodings. As a result, the LERM ensures both prediction discriminability and diversity and can be integrated into existing methods as a plugin. Theoretically, we analyze the relationships between LERM and ERM, as well as between LERM and EntMin. Empirically, we verify the superiority of the LERM under several label insufficient scenarios.

## Installation

We implement SSL and UDA in the Pytorch framework and implement SHDA in the Tensorflow framework.

## Install from Source Code

For SSL and UDA

```shell
pip install -r requirements.txt
```

For SHDA

```shell
XXXX XXXX XXXX XXXX XXXX XXXX

XXXX XXXX XXXX XXXX XXXX XXXX

XXXX XXXX XXXX XXXX XXXX XXXX

XXXX XXXX XXXX XXXX XXXX XXXX

```


## Usage
You can find scripts in the directory `SSL`, `UDA`, and `SHDA`.

## Contact
If you have any problem with our code or have some suggestions, including the future feature, feel free to contact 
- Yulong Zhang (zhangylcse@zju.edu.cn)

or describe it in Issues.


## Acknowledgement

Our implementation is based on the [Transfer Learning Library](https://github.com/thuml/Transfer-Learning-Library), [BNM](https://github.com/cuishuhao/BNM), [SDAT](https://github.com/val-iisc/SDAT).

## Citation
If you find our paper or codebase useful, please consider citing us as:
```latex
@InProceedings{zhang2024rethinking,
  title={Rethinking Guidance Information to Utilize Unlabeled Samples: A Label-Encoding Perspective},
  author={Zhang, Yulong and Yao, Yuan and Chen, Shuhao and Jin, Pengrong and Jin, Jian and Lu Jiangang},
  booktitle={Proceedings of the 41th International Conference on Machine Learning},
  year={2024}
}
