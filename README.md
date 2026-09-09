# Ericsson-Agentic-AI-Compiler
## Project: Agentic AI compiler
Running large AI models is one of the dominant costs in the industry, so every percent of inference or training speedup translates directly into financial gain and larger models. This is why frontier labs invest heavily in model lowering: turning a high-level model definition into fast machine code for a specific accelerator. Today this pipeline mixes manual and automatic stages. At the top, human experts redesign algorithms with mathematical and hardware-aware tricks, such as FlashAttention. Below that, engineers hand-write and tune device kernels (in CUDA, Triton, or Pallas), which is a slow and expertise-heavy task. Compilers then lower the model’s computation graph automatically, performing operator fusion, layout assignment, and scheduling. Compiler is constrained by fixed rewrite rules: it can fuse and schedule operations, yet cannot restructure an algorithm, so on its own it never rediscovers e.g. the streaming-softmax reformulation behind FlashAttention. The kernel-writing stage is an active research area: agentic systems like Sakana AI CUDA Engineer and DeepMind’s AlphaEvolve discover kernels and low-level optimizations that beat expert-tuned baselines. The common thread is that a reasoning agent can read profiles, form hypotheses, rewrite code, and measure the result – the way a human performance engineer does – can explore optimizations that a rule-based compiler cannot.


In this project, an agentic system was built that supplements a compiler to make AI models faster. Given a model or algorithm written in JAX, the array-computation and program-transformation language used by leading AI labs, which runs the same code on different hardware. The agent works in an iterative loop around the XLA compiler: it inspects the lowered program and its runtime profile, proposes high-level rewrites of the JAX code, suggests and implements custom kernels, recompiles, and analyzes the resulting performance to decide the next step. Language models do not always know much about a given accelerator or about compiler behavior, so part of the task was to give the agent the right context and feedback signals: profiling data, compiler intermediate representations (HLO/StableHLO), and correctness checks.

## Project Structure

```text
Ericsson-Agentic-AI-Compiler/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── src/
│   ├── agent/
│   ├── compiler/
│   ├── profiling/
│   └── optimization/
│
└── data/
    └── programs/
```

## Project Setup
### Setup environment
Using Python 3.14.4, setup the venv, activate and verify it
```bash
python3 -m venv .venv
source .venv/bin/activate
python --version
which python
```
The Python executable should point to the .venv directory inside the project.

Then optionally, but recommended to automatically activate the environment when entering the repository, setup direnv using this command for macOS google for other OS
```bash
brew install direnv
```

Create the .envrc 
```bash
touch .envrc
```

Open .envrc and copy-paste this line into the file
```bash
source .venv/bin/activate
```

Then paste this command into terminal
```bash
direnv allow
```

### Install requirements
```bash
pip install -r requirements.txt
```

## Pipeline

### Current small test
This is a small test currently to test that jax works
```bash
python src/compiler/compile.py 
```

It should output something like this in the terminal
```text
=== Lowered program ===
module @jit_simple_program attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<1000x1000xf32>, %arg1: tensor<1000x1000xf32>) -> (tensor<1000x1000xf32> {jax.result_info = "result"}) {
    %0 = stablehlo.sine %arg0 : tensor<1000x1000xf32>
    %cst = stablehlo.constant dense<2.000000e+00> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1000x1000xf32>
    %2 = stablehlo.multiply %arg1, %1 : tensor<1000x1000xf32>
    %3 = stablehlo.add %0, %2 : tensor<1000x1000xf32>
    return %3 : tensor<1000x1000xf32>
  }
}

Result shape: (1000, 1000)
Execution time: 0.952 ms
```

## Contact
John Christensen - johnchristensen@outlook.com


Joel Maharena -    


Lidya Nasser -        


Sara Strandberg -   