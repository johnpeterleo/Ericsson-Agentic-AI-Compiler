# Ericsson-Agentic-AI-Compiler


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