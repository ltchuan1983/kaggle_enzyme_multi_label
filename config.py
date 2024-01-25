{
    "load_data" : {
        "traindata_filepath": "../data/train.csv",
        "all_labels": ["EC1", "EC2", "EC3", "EC4", "EC5", "EC6"],
        "target_labels": ["EC1", "EC2"],
        "labels_to_drop": ["EC3", "EC4", "EC5", "EC6"]
    },

    "features" : {
        "categorical_features": ["fr_COO", "fr_COO2"]
    },

    "BATCH_SIZE" : 32,
    "RANDOM_SEED": 808,
    "CONTAMINATION": 0.08
}