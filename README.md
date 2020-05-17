# wavenet-tf2.0
this project implements a wavenet with tensorflow 2.0

## download dataset

wavenet is trained on VCTK which is available [here](https://datashare.is.ed.ac.uk/handle/10283/2651). unzip the zip file aftern downloading it.

## how to create dataset

create dataset with the following command

```bash
python3 create_dataset.py <path/to/VCTK-Corpus>
```

## how to train

train wavenet with the following command

```bash
python3 train.py
```
