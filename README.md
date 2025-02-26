# ZenQ: AI Simulator for AMD Accelerators

ZenQ is a deep learning framework with optimized implementations of popular AI kernels and a highly accurate simulator for AMD accelerators. It provides a comprehensive profiling tool for the backend and utilizes reinforcement learning to predict the performance of the kernels.

### Outline

1. **Accelerated Backend**: Utilizes AMD ROCm for GPU acceleration and AMD AVX512 for CPU acceleration. Implements kernels commonly used in deep learning.
2. **Profiler**: Provides a comprehensive profiling tool for the backend.
3. **Learnt Simulator**: Utilizes reinforcement learning to learn the behavior of the backend and predict the performance of the kernels.

### Setup `poetry` for Python package management

Download and install `poetry` using:

```bash
curl -sSL https://install.python-poetry.org | python3 -
```

Poetry sometimes bricks. To overcome this problem, use:

```bash
export PYTHON_KEYRING_BACKEND=keyring.backends.null.Keyring
```

Then, install all necessary packages using:

```bash
poetry config virtualenvs.in-project true
poetry install --no-root
```

Finally, activate the built environment:

```bash
poetry shell
```

If you either added or removed packages, then you can save a checkpoint of the `conda` environment by:

```bash
poetry export -f requirements.txt --output requirements.txt --all-extras
poetry lock
```

For additional documentation information, follow [poetry](https://python-poetry.org/docs/#installing-with-the-official-installer) official docs.

### Install the backend

To install the backend, use the following command:

```bash
mkdir -p build && cd build
cmake .. -DHPC_ENABLE_CUDA=ON -DHPC_COLORED_LOGGING=ON -DCMAKE_BUILD_TYPE=Release -DHPC_BUILD_BENCHMARKS=ON; cmake --build . -j$(nproc)
```

### Run the backend

To run the tests, inside `build` directory, use the following command:

```bash
./test_main
```

To run the benchmarks, inside `build` directory, use the following command:

```bash
./bin/run_benchmarks
```

### Notes

<div style="border: 1px solid #ddd; padding: 20px; margin: 20px 0;">

**Example for adding specific pytorch version**:

```bash
poetry source add --priority explicit pytorch_cpu https://download.pytorch.org/whl/cpu
poetry add --source pytorch_cpu torch torchvision
```

</div>
