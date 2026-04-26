# Raw Dataset Files

The dataset-variability experiment expects local CSV files here. Kaggle files
usually need to be downloaded manually from a browser or with Kaggle credentials.

Expected layout:

```text
data/raw/
├── diabetes_prediction/
│   └── diabetes_prediction_dataset.csv
├── cardiovascular/
│   └── cardio_train.csv
├── heart_indicators/
│   └── heart_2020_cleaned.csv
├── diabetes_hospital/
│   └── diabetic_data.csv
├── compas/
│   └── cox-violent-parsed.csv
└── glioma/
    └── TCGA_InfoWithGrade.csv
```

Adult is handled separately because it uses the original UCI train/test files:

```text
data/adult/
├── adult.data
└── adult.test
```

COMPAS can also be downloaded automatically by `data/load_compas.py` when network
access is available, and is cached as `data/compas_raw.csv`. The local
`data/raw/compas/cox-violent-parsed.csv` file is used first when present.

Registered dataset keys:

- `adult`
- `compas`
- `diabetes_prediction`
- `cardiovascular`
- `heart_indicators`
- `diabetes_hospital`
- `glioma`
