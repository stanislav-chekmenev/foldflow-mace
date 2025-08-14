<div align="center">

# Learning and Modifying SE(3) Flow Matching for Protein Generation

</div>

This code is a fork of the original [FoldFlow repository](https://github.com/DreamFold/FoldFlow), building upon their prior work which introduced the first flow matching model for protein backbone generation. **FoldFlow** models are [flow matching](https://github.com/atong01/conditional-flow-matching) generative models for protein design and work by generating protein structures as represented on the $SE(3)^N_0$ manifold. 

The repo serves as a learning step towards my transition to the Bio ML field. The months of diving into the code, modifying it and trying to improve the baseline model have taught me new valuable concepts and ideas in the field of the modern protein discovery. My idea was to enhance the original implementation of the FoldFlow-2 architecture with a $\text{SE}(3)$-equivariant **self-conditioned** GNN encoder based on the [MACE](https://arxiv.org/abs/2206.07697) model. 

![](media/foldflow-mace_protein.gif)

**You can find all the information about my learning adventure in this [post](https://chekmenev.me/posts/protein_discovery/) that sheds light upon the theory and practical implementation of $\text{SE}(3)$ manifold flow matching for protein backbone generation augmented with an additional $\text{SE}(3)$-equivariant GNN encoder.**

<div>


## Installation
The installation follows the steps listed in the original repo.

First, you should [install micromamba](https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html) by running the next command on Linux or MacOS.

```bash
# Install micromamba
"${SHELL}" <(curl -L micro.mamba.pm/install.sh)
```

Then clone the repo and install the dependencies:

```bash
git clone https://github.com/stanislav-chekmenev/foldflow-mace.git
cd foldflow-mace

# Install dependencies and activate environment
micromamba create -f environment.yaml
micromamba activate foldflow-env
```

## Training & Inference 

Training & Inference follow closely the original configurations proposed in the [FoldFlow repo](https://github.com/DreamFold/FoldFlow). SO if you want to train your model, please, check the original repo for details first. I'll just highlight main differences here.

### MACE encoder

Since I added a switchable equivariant GNN encoder to the base architecture of FoldFlow-2 you could turn it off and on, as well as setup its parameters, in the hydra config file: `runner/config/model/ff2_mace.yaml`.

### Training FoldFlow-2

I experimented with FoldFlow-2 model without any RL finetuning part. That means that I slightly modified the `Experiment` class defined in `train.py`, which allowed it to work with FoldFlow-2 architecture, and used the loss defined for FoldFlow-1 for training. My training experiments ran only on the subset of training examples of around 4K PDB proteins of lengths 200 &dash; 300 amino acids.


### Acknowledgements

I'd like to thank the authors of [FoldFlow](https://github.com/DreamFold/FoldFlow) and [FrameDiff](https://github.com/jasonkyuyim/se3_diffusion) papers and codebases for their work!