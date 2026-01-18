# Mind the Gap: Benchmarking LLM Uncertainty, Discrimination, and Calibration in Specialty-Aware Clinical QA

This repository contains the code, data processing scripts, and experiments accompanying the paper:

**Mind the Gap: Benchmarking LLM Uncertainty, Discrimination, and Calibration in Specialty-Aware Clinical QA**  
Alberto Testoni, Iacer Calixto  
[arXiv:2506.10769](https://arxiv.org/abs/2506.10769) - Accepted to EACL 2026 (Main Conference).


---
## Repository Status

⚠️ **Work in Progress**  
This repository is being updated gradually. Some scripts and documentation may be incomplete or subject to change. Please check back regularly for updates.

---

## Overview

Reliable uncertainty quantification (UQ) is critical for deploying large language models (LLMs) in high-stakes domains such as clinical question answering (QA).  
In this work, we systematically benchmark uncertainty estimation methods across:

- **10 open-source LLMs** (general-purpose, biomedical, and reasoning models)  
- **11 clinical specialties** (both common and underrepresented)  
- **6 question types** (diagnosis, treatment, diagnostic test, definition, procedure/operation, other)  

We evaluate score-based methods, sampling-based consistency measures, and set-based conformal prediction, and we introduce a lightweight case study based on behavioral features from reasoning traces.

---

## Citation

If you use this code or data in your research, please cite:

```bibtex
@article{testoni2025mind,
  title={Mind the Gap: Benchmarking LLM Uncertainty, Discrimination, and Calibration in Specialty-Aware Clinical QA},
  author={Testoni, Alberto and Calixto, Iacer},
  journal={arXiv preprint arXiv:2506.10769},
  year={2025}
}
