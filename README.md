# Scaling BERT4Rec for Efficient Recommendation

This project explores the effects of scaling model size and applying system-level optimizations on the training efficiency and accuracy of **BERT4Rec**, a transformer-based sequential recommender system.

We study model variants ranging from 85M to 430M parameters on the **Amazon Electronics** dataset, and evaluate the impact of:
- Mixed-precision training (AMP)
- Graph-level optimization (`torch.compile`)
- Parameter-efficient fine-tuning (iLoRA)
- Hardware differences (NVIDIA A100, V100, L4)

---

## Project Milestones

| Milestone                                        | Status      |
|--------------------------------------------------|-------------|
| Original baseline BERT4Rec implementation (5M model) on MovieLens-20M dataset     | ✔ Complete |
| Scaled-up model variants (85M, 130M, 430M) on Amazon Product (Electronics) dataset      | ✔ Complete |
| Hyperparameter tuning for scaled up variants             | ✔ Complete |
| AMP and `torch.compile` integration              | ✔ Complete |
| iLoRA implementation and integration             | ✔ Complete |
| Experimental benchmarking and profiling          | ✔ Complete |
| Final evaluation and results summarization       | ✔ Complete |

---

## Training and Inference Optimizations

| Optimization                          | Purpose                                                  | Where/How Applied                                                      |
| ------------------------------------- | -------------------------------------------------------- | ---------------------------------------------------------------------- |
| **PyTorch AMP (Mixed Precision)**     | Speeds up training and reduces GPU memory usage          | Enabled via `torch.amp.autocast()` + `GradScaler`                      |
| **torch.compile (reduce-overhead)**   | Removes Python overhead, fuses ops for better throughput | Applied with `torch.compile(model, mode="reduce-overhead")` before DDP |
| **Checkpointing & Resume**            | Avoids re-training in case of interruption               | Supports `--resume_checkpoint` in CLI                                  |
| **Pickle-based Dataset Caching**      | Speeds up repeated data loading                          | Uses `.pkl` caching (planned extension to `preprocess_amazon_data`)    |
| **Gradient Clipping**                 | Prevents exploding gradients                             | `apply_gradient_clipping(model, max_norm=1.0)`                         |
| **Cosine LR Schedule with Warmup**    | Improves convergence and training stability              | Used in `get_cosine_schedule_with_warmup()`                            |
| **Improved Loss (Focal Loss)**        | Stabilizes learning for imbalanced distributions         | Custom `focal_loss_on_masked()`                                        |
| **Parallel Data Loading**             | Speeds up data pipeline                                  | `DataLoader(num_workers=4, pin_memory=True)`                           |
| **Garbage Collection + empty\_cache** | Frees unused memory between epochs                       | Controlled with `--gc_after_epoch`                                     |
| **torch.jit.trace**                   | For JIT tracing optimized inference model                | Used in `--use_torch_jit_trace` mode                                   |


## Repository Structure

```text
├── cache/                  # Contains pickled Amazon Electronics data
├── models/                 # Trained BERT4Rec model variants
│   └── bert4rec_5M.pt      # not added because size > 100MB and GitHub did not allow adding it
├── scripts/                # Training and evaluation scripts
│   ├── baseline5M.py       # Initial baseline
│   ├── baseline_final.py   # Training loop with AMP, compile
│   └── ilora.py            # iLoRA implementation
├── configs/                # YAML config files per model size
│   └── baseline_85M.yaml     
│   └── baseline_130M.yaml      
│   └── baseline_430M.yaml       
├── results/                # Output visualizations
│   └── profiler_outputs/   # PyTorch profiler outputs
├── requirements.txt        # Required dependencies
└── README.md               # This file
```

---

## Example Commands

To run on a single GPU:
```
torchrun --nproc_per_node=1 scripts/baseline.py --config configs/bert4rec_130m.yaml --use_amp --use_torch_compile --run_profiler --run_benchmark
```

To run on 2 GPUs:
```
torchrun --nproc_per_node=2 --distributed scripts/baseline.py --config configs/bert4rec_130m.yaml --use_amp --use_torch_compile --run_profiler --run_benchmark
```

## Results and Observations

* Accuracy vs. Model Size

Original baseline results (MovieLens-20M):
Model Size: 5M,   HR\@10: 0.0621,   NDCG\@10: 0.0458 


Final Results (Amazon Electronics):
| Model Size | HR\@10 | NDCG\@10 |
| ---------- | ------ | -------- |
| 85M        | 0.0818 | 0.0637   |         
| 130M       | 0.0855 | 0.0682   |
| 430M       | 0.0858 | 0.0686   |

Observation: Accuracy improves with model size.

* Training Throughput (Samples/sec)

| Model + Config       | Throughput | Epoch Time |
| -------------------- | ---------- | ---------- |
| 130M + AMP           | 11,428     | 7.5 min    |
| 130M + AMP + Compile | 13,300     | 6.8 min    |
| 130M + AMP + iLoRA   | 10,800     | 7.9 min    |

Observation: torch.compile improves throughput by ~17%, and AMP nearly doubles it. iLoRA slightly reduces throughput due to adapter overhead but enables massive parameter reduction (0.55% trainable).

* Memory Usage (Peak GPU Memory)

| Model + Config     | Memory (MB) |
| ------------------ | ----------- |
| 130M Baseline      | 11,000      |
| 130M + AMP         | 7,200       |
| 130M + AMP + iLoRA | 6,400       |

Observation: AMP and iLoRA significantly reduce memory footprint, enabling larger batch sizes.


## Results and Observations
We evaluated the impact of model size and system-level optimizations on BERT4Rec training throughput, accuracy, and efficiency, including a study on instance-wise LoRA (iLoRA). Our metrics include:

* Step Time (ms): Total time per training step

* Self CUDA Time (ms): GPU kernel execution time

* Peak Memory Usage (GB): Maximum CUDA memory consumed


### Optimization Comparison Summary

| Config                          | Step Time     | Self CUDA Time | Peak Memory   |
| ------------------------------- | ------------- | -------------- | ------------- |
| Baseline (No AMP/Compile)       | 278.4 ms      | 201.2 ms       | 11.2 GB       |
| AMP                             | 280.5 ms      | 204.4 ms       | 7.2 GB        |
| Compile                         | 284.8 ms      | 203.5 ms       | 11.2 GB       |
| AMP + Compile                   | **58.3 ms**  | **32.6 ms**   | **7.2 GB**   |
| iLoRA (No AMP/Compile)          | 470 ms        | 470 ms         | 10.4 GB       |
| iLoRA + AMP                     | 173 ms        | 100.7 ms       | 2.9 GB        |
| iLoRA + AMP + Compile           | 975 ms       | 908 ms        | 8.1 GB        |
| iLoRA + AMP + Compile (Reduced) | **130 ms**   | **76.8 ms**   | **2.85 GB**  |


* Hardware Comparison: NVIDIA A100, V100, L4

![Hardware](https://github.com/mahi397/Optimizing-Transformer-based-Sequential-RecommenderSystems/blob/main/results/epoch_time_vs_hardware.png)


* Performance vs Model Size

![Perf](https://github.com/mahi397/Optimizing-Transformer-based-Sequential-RecommenderSystems/blob/main/results/results/hr_vs_model_size.png)


* Training efficiency vs Model size

![Throughput](https://github.com/mahi397/Optimizing-Transformer-based-Sequential-RecommenderSystems/blob/main/results/throughput_vs_model_size.png)

![Time](https://github.com/mahi397/Optimizing-Transformer-based-Sequential-RecommenderSystems/blob/main/results/time_vs_model_size.png)


* Profiling Insights: 130M Model

| Metric                   | No Opt     | AMP Only   | Compile Only | AMP + Compile           |
| ------------------------ | ---------- | ---------- | ------------ | ----------------------- |
| **Self CUDA Time Total** | 201.2 ms   | 204.4 ms   | 203.5 ms     | **32.6 ms**            |
| **Step Time** (approx)   | \~278 ms   | \~280 ms   | \~284 ms     | **\~58 ms**            |
| **Peak CUDA Memory**     | \~11.2 GB  | \~7.2 GB  | \~11.2 GB    | \~7.2 GB               |
| **Top Op by CUDA %**     | `aten::mm` | `aten::mm` | `aten::mm`   | `CompiledFunction + mm` |
| **Launch Overhead**      | High       | High       | Lower        |  Lowest                |


![130Mcudatime](https://github.com/mahi397/Optimizing-Transformer-based-Sequential-RecommenderSystems/blob/main/results/Self_CUDA_Time.png)

![130Mcudamem](https://github.com/mahi397/Optimizing-Transformer-based-Sequential-RecommenderSystems/blob/main/results/Peak_CUDA_Memory.png)


* Profiling Insights: iLoRA

| Config                            | Step Time      | Self CUDA   | Memory        |
| --------------------------------- | -------------- | ----------- | ------------- |
| iLoRA                             | \~470 ms       | \~470 ms    | 10.4 GB       |
| iLoRA + AMP                       | \~173 ms       | 100.7 ms    | 2.9 GB        |
| iLoRA + AMP + Compile             | \~975 ms       | 908 ms      | 8.1 GB        |
| iLoRA + AMP + Compile (Reduced) | **\~130 ms**  | **76 ms**  | **2.85 GB**  |


![iLoRA](https://github.com/mahi397/Optimizing-Transformer-based-Sequential-RecommenderSystems/blob/main/results/iLoRA.png)

