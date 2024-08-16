## Environment Setup
Install [graph-tool](https://graph-tool.skewed.de/) and activate the Conda environment:
```bash
conda create --name gt -c conda-forge graph-tool
conda activate gt
```

Install the additional required Python packages:
```bash
pip install networkx
pip install scikit-learn
```

For using the algorithm based on Integrer Linear Programming, install Gurobi:
```bash
pip install gurobipy
```

Please note that the Gurobi package includes a trial license for solving problems of limited size. For instructions on obtaining and activating a full product license, refer to the [Gurobi Installation Guide](https://www.gurobi.com/documentation/quickstart.html).

For using pysbm-based algorithms, clone the pysbm repository:
```bash
git clone https://github.com/funket/pysbm.git
```

For running experiments on brain networks, install nilearn:
```bash
pip install nilearn
```
