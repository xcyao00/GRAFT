# Back to the Definition: Estimating Step-Level Advantages via Trajectory Graphs for Agentic Reinforcement Learning

This work proposes GRFAT, which is a graph-augmented reinforcement learning algorithm for agent training. It is based on building a state-transition graph from rollout trajectories and using graph-based value estimates to assign step-level advantages, providing denser and more informative credit signals.

## Key Idea

During each training iteration, GRFAT:

1. Collects multi-turn trajectories via the agent-environment loop.
2. Builds a `MultiSinkValueGraph` per task group from the observed state transitions.
3. Computes per-node values via Bellman iteration.
4. Assigns per-step advantages as TD residuals `γ·V(s') − V(s)`.
5. Optionally applies GAE on top of the graph-based value function for smoother advantage estimates.
6. Normalizes step-level advantages within same-state groups and optionally combines with episode-level advantages.

## Configuration

GRFAT has its own trainer entry point (`recipe.graft.main_graft`) and Hydra config (`recipe/graft/config/graft_trainer.yaml`). Set `algorithm.adv_estimator=graft` to activate it.

### Core algorithm parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `algorithm.adv_estimator` | `graft` | Must be set to `graft`. |
| `algorithm.gamma` | `0.99` | Decay factor. Used in Bellman iteration. |
| `algorithm.graft.step_advantage_w` | `1.0` | Weight on the graph-derived step-level advantage. |
| `algorithm.graft.episode_advantage_w` | `0.0` | Weight on the episode-level outcome advantage. Set to `0.0` to rely solely on the graph step signal. Ignored when `gae_mode != none`. |
| `algorithm.graft.mode` | `mean_std_norm` | Normalization mode for step and episode advantages. `mean_std_norm`: subtract mean and divide by std. `mean_norm`: subtract mean only. |
| `algorithm.graft.advantage_norm_scope` | `node` | Grouping scope for step-level advantage normalization. `node`: group = (task uid, anchor_obs) — steps compete only against others at the same graph node (original behaviour; singleton nodes get zeroed). `task`: group = task uid only — all steps of all trajectories in the same task pooled together, avoiding singleton-zeroing at the cost of cross-state comparison. |
| `algorithm.graft.enable_similarity` | `False` | If `True`, merge observations with similarity ≥ `similarity_thresh` into a single graph node, reducing fragmentation from minor text differences. |
| `algorithm.graft.similarity_thresh` | `0.9` | String similarity threshold used when `enable_similarity=True`. |


### GAE-based step advantage parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `algorithm.graft.gae_mode` | `none` | GAE scheme. `none`: keep TD(0) + episode advantage pipeline.  `gae_graph_multihop`: multi-hop edge-global GAE; TD(k) for k≥1 is the uniform mean over all raw edge instances reachable in exactly k hops. |
| `algorithm.graft.gae_lambda` | `0.95` | Lambda for GAE modes. Ignored when `gae_mode=none`. |

### Graph data saving parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `algorithm.graft.graph_save_freq` | `0` | Save graph nodes/edges/V-values to JSON every N training steps. `0` = disabled. Files written to `<graph_save_dir>/step_<N>.json`. |
| `algorithm.graft.graph_save_dir` | `""` | Directory for graph JSON files. Empty string defaults to `<trainer.default_local_dir>/graph_data/`. |


## Install Packages

> Our experiments were run on Ascend 910B, based on vllm=0.13.0. The environment can be successfully configured using the following instructions. If you have your own vllm runtime environment, you do not need to configure the vllm environment according to the following instructions.


```bash
apt-get update -y && apt-get install -y gcc g++ cmake libnuma-dev wget git curl jq
pip config set global.extra-index-url "https://download.pytorch.org/whl/cpu/"
pip install vllm==0.13.0
pip install vllm-ascend==0.13.0

pip uninstall -y triton triton-ascend
pip install triton-ascend==3.2.0
```

## Install Environments

Note that we are built on the `verl-agent` code library. If there are any issues with environment configuration, you can refer to the environment configuration instructions in the [verl-agent](https://github.com/langfengQ/verl-agent) library.

### 1. ALFWorld
Install with pip:
```bash
pip3 install gymnasium==0.29.1
pip3 install stable-baselines3==2.6.0
pip install alfworld
```

Download PDDL & Game files and pre-trained MaskRCNN detector (will be stored in `~/.cache/alfworld/`):
```bash
alfworld-download -f
```

Play a Textworld game:
```bash
alfworld-play-tw
```

---

### 2. WebShop

Install WebShop:
```bash
cd ./agent_system/environments/env_package/webshop/webshop
./setup.sh -d all
```

Note: If you encounter issues with gdown, you may need to visit `https://drive.google.com/`, get your Google Drive cookie, and paste it into `.cache/gdown/cookies.txt`.
Or you may need to manually download the files.

After WebShop is installed, return to the root directory of the repository and install the verl package:
```bash
cd repo_root/
pip3 install -e .
```

---

### 3. Search
```bash
cd ./agent_system/environments/env_package/search/third_party
pip install -e .
pip install gym==0.26.2
```

Prepare dataset (data will be saved at `~/data/searchR1_processed_direct`):
```bash
cd repo_root/
python examples/data_preprocess/preprocess_search_r1_dataset.py
```


Since faiss-gpu is not available via pip, we setup a separate conda environment for the local retrieval server. Running this server will use around 6GB of GPU memory per GPU, so make sure to account for this in your training run configuration. Build Retriever environments:
```bash
# Create and activate the retriever environment with Python 3.10
conda create -n retriever python=3.10 -y
conda activate retriever

# Install PyTorch (with GPU support) and related libraries
conda install numpy==1.26.4 # needed to stop incompatible version of numpy from being installed via pip
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124

# Install other Python packages
pip install transformers datasets pyserini huggingface_hub

# Install the GPU version of faiss
conda install faiss-gpu==1.8.0 -c pytorch -c nvidia -y

# Install the API service framework
pip install uvicorn fastapi
```

Download the index:
```bash
conda activate retriever

local_dir=~/data/searchR1
python examples/search/searchr1_download.py --local_dir $local_dir
cat $local_dir/part_* > $local_dir/e5_Flat.index
gzip -d $local_dir/wiki-18.jsonl.gz
```

Start the local flat e5 retrieval server: 
```bash
conda activate retriever

# redirect the output to a file to avoid cluttering the terminal
# we have observed outputting to the terminal causing spikes in server response times
bash examples/search/retriever/retrieval_launch.sh > retrieval_server.log 
```


## Quick Start

> Our implementation is built on the verl-agent code library. To run successfully in our environment, some files may have been modified. If you encounter any problems during the running process, you can compare the differences between the original code in verl-agent library and our code, and set a runtime code that fits your environment.

### ALFWorld — Qwen2.5-1.5B

```bash
bash recipe/graft/run_alfworld_1.5b.sh
```

### ALFWorld — Qwen2.5-7B

```bash
bash recipe/graft/run_alfworld_7b.sh
```

### WebShop — Qwen2.5-1.5B

```bash
bash recipe/graft/run_webshop_1.5b.sh
```

### WebShop — Qwen2.5-7B

```bash
bash recipe/graft/run_webshop_7b.sh
```

### SearchQA — Qwen2.5-3B

```bash
bash recipe/graft/run_search_3b.sh
```

### SearchQA — Qwen2.5-7B

```bash
bash recipe/graft/run_search_7b.sh
```

## Visualization

### `visualize_task.py` — single task graph viewer

Quick interactive viewer that prints all tasks in a batch and saves a PNG graph for the selected one.

```bash
python -m recipe.graft.visualize_task <step_N.json> [uid_or_index]
```

## Citation

If you find this work helpful, please cite us.

```bibtex
@article{yao2026graft,
  title = {Back to the Definition: Estimating Step-Level Advantages via Trajectory Graphs for Agentic Reinforcement Learning},
  author = {Xincheng Yao and Haobo Fu and Weiming Liu and Chongyang Zhang},
  journal = {arXiv preprint arXiv:2609.28963},
  year = {2026},
}
```

## Acknowledge

Codes implementations are based on [verl-agent](https://github.com/langfengQ/verl-agent). Thanks for their great contributions!
