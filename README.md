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
This repo is evolving for my phd thesis and for the accepted version of the paper see
- hifuku: `tro-accepted` tag for Domain 4 and `rebuttal` branch for Domain 1-3.
- rpbench: `tro-resubmit` tag for Domain 4 and `rebuttal` branch for Domain 1-3.
- plainmp: `tro-resubmit` tag for Domain 4
- scikit-motionplan: `tro-submit tag for Domain 1-3
```bibtex
@article{ishida2024coverlib,
  title={CoverLib: Classifiers-equipped Experience Library by Iterative Problem Distribution Coverage Maximization for Domain-tuned Motion Planning},
  author={Ishida, Hirokazu and Hiraoka, Naoki and Okada, Kei and Inaba, Masayuki},
  journal={IEEE Transactions on Robotics (T-RO)},
  year={2024}
}
```
