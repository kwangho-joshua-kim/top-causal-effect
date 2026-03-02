# Topological Causal Effects

This repository contains a Python implementation of the methods in the paper
“Topological Causal Effects”.

The code reproduces the experiments on three datasets:
- SARS-CoV-2 (semi-synthetic)
- GEOM-Drugs (semi-synthetic)
- ORBIT (synthetic)

# Setup
Create a virtual environment and download all necessary packages using the following command:
```
conda env create -n topo_causal --file environment.yaml
```
Activate virtual environment:
```
conda activate topo_causal
```

# SARS-CoV-2
Run the **SARS_COV2/main.ipynb** file.


# GEOM-Drugs
Download *val_data.pickle* from [here](https://bits.csb.pitt.edu/files/geom_raw/) and place it in the *GEOM-Drugs/data* directory. Then, run the **GEOM-Drug/main.ipynb** file.

# ORBIT
To create the dataset, move to the *ORBIT* directory and run:
```
python generate_data.py 
```
Then, run the **ORBIT/main.ipynb** file.