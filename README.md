## installation

```bash
sudo apt-get install libgoogle-glog-dev  # voxbloxpy
```
```bash
git clone https://github.com/iory/scikit-robot.git
cd scikit-robot
git remote add h-ishida https://github.com/HiroIshida/scikit-robot.git
git fetch h-ishida
git checkout h-ishida/master
```

Then
```
pip3 install -e .
```

## knowledges

### joint-training of encoder with fcn vs only fcn
At least for "ClutteredFridge_SQP" domain (2000, 80), training only fcn fixing encoder is better than others
https://gist.github.com/HiroIshida/18c4dc62e284267a9c000a604ca4671a
https://gist.github.com/HiroIshida/7c40f870fe48c6e1aa00aa47b8451baa

### iterpred training will be much more efficient with batchnorm
For "ClutteredFridge_SQP" domain (2000, 80), training with batch norm outperform the counterpart in terms of learning speed and obtained loss
```
[INFO] 2024-02-12 06:36:44,295 root: elapsed time: 117.2144844532013
[INFO] 2024-02-12 06:36:44,295 root: iter 0: with batch norm => loss 2.059505508840084
[INFO] 2024-02-12 06:42:54,419 root: elapsed time: 354.125629901886
[INFO] 2024-02-12 06:42:54,419 root: iter 0: without batch norm => loss 2.187796725332737
[INFO] 2024-02-12 06:51:00,699 root: elapsed time: 117.24607849121094
[INFO] 2024-02-12 06:51:00,699 root: iter 1: with batch norm => loss 2.160003073513508
[INFO] 2024-02-12 06:55:05,974 root: elapsed time: 228.61053204536438
[INFO] 2024-02-12 06:55:05,974 root: iter 1: without batch norm => loss 2.3470986500382423
[INFO] 2024-02-12 07:03:27,924 root: elapsed time: 128.58032202720642
[INFO] 2024-02-12 07:03:27,924 root: iter 2: with batch norm => loss 1.9947637483477592
[INFO] 2024-02-12 07:06:33,240 root: elapsed time: 166.7109591960907
[INFO] 2024-02-12 07:06:33,240 root: iter 2: without batch norm => loss 2.0556017830967903
[INFO] 2024-02-12 07:14:57,482 root: elapsed time: 132.91688585281372
[INFO] 2024-02-12 07:14:57,482 root: iter 3: with batch norm => loss 1.44915728867054
[INFO] 2024-02-12 07:20:30,565 root: elapsed time: 320.5342843532562
[INFO] 2024-02-12 07:20:30,565 root: iter 3: without batch norm => loss 1.6301937445998191
```
