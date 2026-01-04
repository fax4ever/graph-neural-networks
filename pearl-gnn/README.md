# Main project

```bash
uv sync
```

```bash
uv run script/main_train.py
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

## *Alternative* Visualize dataset: Deep Learning Sapienza Hackaton dataset

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