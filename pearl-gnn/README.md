# Main project

@misc{kanatsoulis2025learningefficientpositionalencodings,
      title={Learning Efficient Positional Encodings with Graph Neural Networks}, 
      author={Charilaos I. Kanatsoulis and Evelyn Choi and Stephanie Jegelka and Jure Leskovec and Alejandro Ribeiro},
      year={2025},
      eprint={2502.01122},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2502.01122}, 
}

Reimplementation of PEARL. Tested only on ZINC dataset.

```bash
uv sync
```

```bash
uv run script/main_train.py
```

## Slides

[PEARL & Positional Encodings in Graph Neural Networks.pptx](PEARL%20%26%20Positional%20Encodings%20in%20Graph%20Neural%20Networks.pptx)


# Alternative installation: ROCm + conda

conda create -n pearl-pe python=3.13
conda activate pearl-pe
pip install torch pytorch-triton-rocm --index-url https://download.pytorch.org/whl/rocm6.2.4
pip install numpy matplotlib networkx torch-geometric ipykernel
conda install -n pearl-pe ipywidgets -y
conda run -n pearl-pe jupyter nbextension enable --py widgetsnbextension --sys-prefix
conda install conda-forge::pytest
pip install -e . --no-deps  # Install only the package code in editable mode (no dependencies)
pytest
python script/main_zinc.py

## Verify ROCm

```python
import sys, torch

> sys.version # '3.13.11 | packaged by conda-forge | (main, Dec  6 2025, 11:24:03) [GCC 14.3.0]'
> torch.cuda.is_available() # True
> torch.cuda.get_device_name(0) # 'AMD Radeon RX 7800 XT'
```

# Visualize datasets

Not part of the reproducer.

## Visualize dataset: ZINC dataset from torch_geometric

```bash
uv sync
```

```bash
uv run script/main_zinc.py
```

```bash
uv run pytest
```

## *Alternative* Visualize dataset: Deep Learning Sapienza Hackathon dataset

Download the dataset and copy the folders (A/B/C/D) at the root of this subproject (graph-neural-networks/pearl-gnn):

https://drive.google.com/drive/folders/1Z-1JkPJ6q4C6jX4brvq1VRbJH5RPUCAk

```bash
uv sync
```

```bash
uv run script/main_deepl.py
```

### More info about the Deep Learning Sapienza Hackaton

https://sites.google.com/view/hackathongraphnoisylabels/home
https://huggingface.co/spaces/examhackaton/GraphClassificationNoisyLabels