# MIMIC-DL2

This project uses the [DL2 framework](https://github.com/eth-sri/dl2) to create logical constraints for the in-hospital-mortality benchmark dataset for the [MIMIC-III Database](https://physionet.org/content/mimiciii/1.4/). See following paper for the MIMIC-III benchmarks:
[Multitask learning and benchmarking with clinical time series data](https://arxiv.org/abs/1703.07771).

## Installation
The code has been tested on machines running python >= 3.8.
To install and run the code, it is assumed that one has access to the following:

- Our fork of the [DL2 framework](https://github.com/IvoAA/dl2) - OBS. it must be the master branch. 
- The [MIMIC-III Benchmarks Code](https://github.com/YerevaNN/mimic3-benchmarks). 
- The in-hospital-mortality dataset placed inside the 'data' folder.

The former two should simply be cloned locally, and then the locations must be added to the PYTHONPATH.

To recreate the in-hospital-mortality dataset, one must have access to the MIMIC-III Database. Then, the benchmark can be created by following the instruction outlined in the repository for the MIMIC-III benchmarks as given above.

The remaining dependencies are specified in the [requirements file](requirements.txt).

## Usage
To run the code, ...
