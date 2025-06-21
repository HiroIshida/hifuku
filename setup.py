from setuptools import find_packages, setup

setup_requires = []

install_requires = [
    "pyinstrument",
    "scikit-motionplan",
    "rpbench",
    "mohou",
    "threadpoolctl",
    "cmaes",
    "numba",
    "psutils",
]

setup(
    name="hifuku",
    version="0.0.4",
    description="experimental",
    author="Hirokazu Ishida",
    author_email="h-ishida@jsk.imi.i.u-tokyo.ac.jp",
    license="MIT",
    install_requires=install_requires,
    packages=find_packages(exclude=("tests", "docs")),
)
