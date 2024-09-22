# config.py
CONFIG = {
    "model_name": "microsoft/DialoGPT-large",
    "dataset_path": "concatenated_data.csv",
    "use_cols": [1, 2],  # Indices for the columns to be used as Questions and Answers
    "max_length": 1000,
    "temperature": 0.7,
    "top_k": 50,
    "top_p": 0.9,
    "eos_token_id": None,  # This will be set dynamically in the main script
    "model_save_path": "finetuned_model"
}
