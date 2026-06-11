# iGUIDE Climate-Crop

**Phenology-Aware Meta-Learning for Geographically Generalizable Crop Yield Prediction**

Shourya Mehta | CS + GIS, University of Illinois Urbana-Champaign | svm5@illinois.edu  
In collaboration with Dr. Anand Padmanabhan, NSF I-GUIDE Institute

---

## Overview

Crop yield models fail to generalize across geographic regions — a critical barrier for global food security applications. This project proposes a **Climate-Conditioned Phenological Adapter (CCPA)**, a lightweight meta-learning module that conditions a frozen geospatial foundation model on regional climate signatures, enabling generalization across agro-ecological zones.

We evaluate under a rigorous **Leave-One-Cluster-Out (LOCO)** protocol across USDA Farm Resource Regions, directly targeting the OOD failure documented by Chakravarty (TMLR 2026) and the negative R² benchmark of Adjei et al. (2025).

## Key Contributions

- Climate-conditioned adapter that dynamically adjusts phenological priors based on target region climate
- MAML-based meta-learning framework treating agro-ecological zones as distinct tasks
- First architectural fix for geographic distribution shift in crop yield prediction
- Rigorous LOCO evaluation across all USDA Farm Resource Regions

## Repository Structure

    .
    data/               Data download scripts (USDA, WRF-HRRR, Sentinel-2)
    models/             CCPA architecture components
        climate_conditioner.py
        phenological_adapter.py
        ccpa.py
        baselines.py
    training/           MAML training loop and LOCO evaluation
        maml_trainer.py
        loco_eval.py
    notebooks/          Jupyter notebooks (pipeline, baselines, training, eval)
    results/            LOCO evaluation results

## Dataset

Built on CropNet (Lin et al., KDD 2024) — https://huggingface.co/datasets/CropNet/CropNet
- 2,200+ U.S. counties, 2017-2022
- Sentinel-2 NDVI imagery + WRF-HRRR weather + USDA county yield records
- Climate conditioning via ERA5 reanalysis variables

## References

- Chakravarty (TMLR 2026): Systematic OOD evaluation across USDA Farm Resource Regions
- Adjei et al. (2025): Africa LOCO benchmark with frozen Prithvi-EO-1.0 embeddings
- Hasan et al. / VITA (2025): Climate-robust pretraining with seasonality-aware sinusoidal prior
- Lin et al. / CropNet (KDD 2024): Dataset
