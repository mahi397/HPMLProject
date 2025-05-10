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
| Original baseline BERT4Rec implementation (5M model) on MovieLens-20M dataset     | ✅ Complete |
| Scaled-up model variants (85M, 130M, 430M) on Amazon Product (Electronics) dataset      | ✅ Complete |
| Hyperparameter tuning for scaled up variants             | ✅ Complete |
| AMP and `torch.compile` integration              | ✅ Complete |
| iLoRA implementation and integration             | ✅ Complete |
| Experimental benchmarking and profiling          | ✅ Complete |
| Final evaluation and results summarization       | ✅ Complete |

---

## Repository Structure

```text
├── data/                    # Preprocessed Amazon Electronics data
├── models/                 # BERT4Rec model variants
│   └── bert4rec5M.pt    # iLoRA module definition
│   └── baseline_85M.pt    # iLoRA module definition
│   └── scaledup_130M.pt    # iLoRA module definition
│   └── scaledup_430M.pt    # iLoRA module definition
│   └── ilora.pt    # iLoRA module definition
├── scripts/                # Training and evaluation scripts
│   ├── baseline.py            # Training loop with AMP, compile, iLoRA
│   └── ilora.py         # HR@10, NDCG@10 evaluation
├── configs/                # YAML/JSON config files per model size
├── wandb/                  # (Optional) Weights & Biases logs
├── tensorboard/                  # (Optional) Weights & Biases logs   
├── results/                # Output plots and tables
└── README.md               # This file
```

---

## Example Commands

To run on a single GPU:
```
torchrun --nproc_per_node=1 scripts/baseline.py --config configs/bert4rec_130m.yaml --use_amp --use_torch_compile --run_profiler
```

To run on 2 GPUs:
```
torchrun --nproc_per_node=2 --distributed scripts/baseline.py --config configs/bert4rec_130m.yaml --use_amp --use_torch_compile --run_profiler
```

## Results and Observations

Original baseline results:
| Model Size: 5M | HR\@10: 0.0621 | NDCG\@10: 0.0458 |

| Model Size | HR\@10 | NDCG\@10 | Throughput (samples/sec) | Time per epoch (minutes) | Memory Usage |
| ---------- | ------ | -------- | ------------------------ | ------------------------ |
| 85M        | 0.0818 | 0.0637   |         
| 130M       | 0.0855 | 0.0682   |
| 430M       | 0.0858 | 0.0686   |
