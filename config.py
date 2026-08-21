"""Single editable settings file for the reproduction repository."""

CONFIG = {
    # "2a" or "2b"
    "dataset": "2a",

    # main | augmentation | srbandmix_variants | xie | architecture
    "experiment": "main",

    # Paths to the official GDF files and true-label MAT files.
    "data_root": "/path/to/BCICIV_2a_gdf",
    "label_root": "/path/to/true_labels_2a",
    "output_root": "results",

    # Paper protocol.
    "subjects": [1, 2, 3, 4, 5, 6, 7, 8, 9],
    "seeds": [0, 1, 2, 3, 4],

    # Leave empty to use the default modes for the chosen experiment.
    # Examples: ["SR_ONLY"], ["FULL", "WO_SINCNET"], ["FULL_8_30"]
    "modes": [],
}
