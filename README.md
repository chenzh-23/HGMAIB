# HGMAIB
## Step 1: Data Preprocessing
Run the `preprocess.py` script to prepare the heterogeneous network for model input. This step generates the necessary node features, adjacency matrices, and other inputs required for training and prediction.
## Step 2: Meta-Graph Optimization
Execute the `train_search.py` script to search for the optimal adaptive meta-graph. The script automatically identifies the best combination of meta-paths to improve drug–target interaction prediction performance.
## Step 3: DTI Prediction
Use the `train.py` script to perform drug–target interaction (DTI) prediction based on the selected adaptive meta-graph. This script applies the optimal meta-graph obtained from the previous search stage to generate 
predictions and output results. Following these steps sequentially ensures reproducibility of the results reported in our study. For further details on parameter settings or data preparation, please refer to the accompanying code documentation and instructions.

## Requirments
```python
python = 3.8 
pytorch = 1.11+cu113 
numpy = 1.22.4 
pandas = 2.0.3
scipy = 1.10.1
scikit-learn = 1.3.2
```
