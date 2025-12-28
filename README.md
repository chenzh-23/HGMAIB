# HGMAIB
## Requirments
```python
python = 3.8 
pytorch = 1.11+cu113 
numpy = 1.22.4 
pandas = 2.0.3
scipy = 1.10.1
scikit-learn = 1.3.2
```
## How to use

1.Preprocess Data Run the `preprocess.py` script to preprocess the raw datasets and construct the heterogeneous network.

2.Train and Evaluate Run the `train.py` script to train the HGMAIB model on the selected meta-graph, perform drug–target interaction prediction, and obtain the results with evaluation metrics.

### Note on Architecture Search 
Following the strategy in AMGDTI, we employ a lightweight proxy model (implemented in train_search.py) to efficiently search for optimal meta-paths. This proxy model uses a simplified backbone to reduce computational overhead during the search phase.

The discovered optimal paths are then transferred to the full HGMAIB model (implemented in train.py), which incorporates the Hierarchical Gated Multi-Head Attention and Information Bottleneck modules for final prediction.

For reproducibility and ease of use, the best-found architectures reported in the paper are hardcoded in train.py by default. Users do not need to run the search process manually to reproduce the main results.
