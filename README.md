#  Profiling Transformer-Based Model

The objective of this project is to investigate the scalability of transformer-based model in multi-gpus environment by tracing the CUDA interface calling via NVProf.

## Command 
The command for recording the running activities is
```bash
nvprof --csv --log-file profiler_output.csv --print-gpu-trace python $TRAINING_SCRIPT.py
```