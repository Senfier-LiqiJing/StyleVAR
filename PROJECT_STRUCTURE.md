# StyleVAR Project Structure

## Directory Organization

```
PML-Project/
├── assets/                      # Image assets for documentation
│   ├── framework.png           # Architecture diagram (99 KB)
│   └── sample.png              # Qualitative results (4.3 MB)
│
├── models/                     # Model architecture implementations
├── utils/                      # Utility functions and helpers
├── eval/                       # Evaluation scripts and metrics
│
├── proposal/                   # Project proposal documents
├── midterm report/             # Midterm progress report
├── reference/                  # Reference papers and materials
├── VAR-README/                 # Original VAR documentation
│
├── README.md                   # Main project documentation (Academic)
├── Style_VAR_final_report.pdf # Final academic report (7.1 MB)
│
├── fine_tune.py               # Fine-tuning script
├── fine_tuner.py              # Fine-tuning trainer class
├── train.py                   # Training script
├── trainer.py                 # Training utilities
├── dist.py                    # Distributed training utilities
├── split_data.py              # Data splitting utilities
├── extract_clean_ckpt.py      # Checkpoint extraction
│
├── demo_sample.ipynb          # Sampling demonstration notebook
├── demo_zero_shot_edit.ipynb  # Zero-shot editing demo
│
├── requirements.txt           # Python dependencies
├── LICENSE                    # Apache 2.0 License
└── .gitignore                 # Git ignore rules
```

## Key Documentation Files

### README.md
Comprehensive academic documentation including:
- Abstract and methodology
- Quantitative evaluation (StyleVAR vs. AdaIN)
- Training and inference instructions
- Future work (GRPO integration)
- Project roadmap with task checklist

### Visual Assets
- **framework.png**: Architectural diagram showing blended cross-attention mechanism
- **sample.png**: Qualitative style transfer results demonstrating texture transfer

## Recent Updates (2026-01-29)

1. ✅ **Fixed LaTeX rendering issue**: Changed `r^{<k}` to `r^{1:k-1}` for proper display
2. ✅ **Organized assets**: Created `assets/` directory for all images
3. ✅ **Updated image references**: All README images now reference `assets/` folder
4. ✅ **Academic refinement**: Enhanced README with formal language and structure
5. ✅ **Added evaluation metrics**: Comprehensive comparison table with AdaIN baseline
6. ✅ **Future work section**: Detailed GRPO and controllability plans
7. ✅ **Project roadmap**: Complete checklist of completed/pending tasks

## Quick Start

See [README.md](README.md) for detailed installation, training, and inference instructions.
