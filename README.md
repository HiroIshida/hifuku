## Installation
Install scikit-robot
```bash
sudo apt update
sudo apt-get install libspatialindex-dev freeglut3-dev libsuitesparse-dev libblas-dev liblapack-dev -y
pip install scikit-robot
```

install build tool
```bash
sudo apt-get install cmake libeigen3-dev -y
sudo apt install libboost-all-dev -y
pip install scikit-build
```

Install `scikit-motionplan`, `plainmp`, `rpbench` and `hifuku` (this repo)
```
pip install git+https://github.com/HiroIshida/scikit-motionplan.git -v
pip install git+https://github.com/HiroIshida/plainmp.git -v
pip install git+https://github.com/HiroIshida/rpbench.git -v
pip install git+https://github.com/HiroIshida/hifuku.git -v
```

## Reference
This repo and relevant packages (`scikit-motionplan`, `rpbench`) are implemented for the following paper [https://arxiv.org/abs/2405.02968](https://arxiv.org/abs/2405.02968).
For the specific commit at the timing of the paper submission, please refer to the tag `tro-submitted` of each repo.
```bibtex
@article{ishida2024coverlib,
  title={CoverLib: Classifiers-equipped Experience Library by Iterative Problem Distribution Coverage Maximization for Domain-tuned Motion Planning},
  author={Ishida, Hirokazu and Hiraoka, Naoki and Okada, Kei and Inaba, Masayuki},
  journal={arXiv preprint arXiv:2405.02968},
  year={2024}
}
```
