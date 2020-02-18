

This code also uses code from the following papers

BatchBALD: Efficient and Diverse Batch Acquisition for Deep Bayesian Active Learning
Bayesian Batch Active Learning as Sparse Subset Approximation

Make sure you install all requirements using

```
conda install pytorch torchvision cudatoolkit=10.0 -c pytorch
pip install -r requirements.txt
```

The acquisition function implementations are in src/multi_bald.py. The regular ICAL is implemented as compute_ical and ICAL-pointwise is implemented as compute_ical_pointwise. Some sample commands are below.
