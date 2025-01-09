import pandas as pd

def load_and_preprocess_data():
    """
    Loads and preprocesses the mental health FAQ and MedQuAD datasets.

    This function reads two CSV files containing questions and answers related to mental health
    and medical queries. It performs preprocessing steps such as:
    - Dropping unnecessary columns from both datasets.
    - Renaming columns in the MedQuAD dataset for consistency.
    - Concatenating the questions and answers from both datasets into a single text string.

    Returns:
    - text_combined (str): A single string containing the concatenated questions and answers from 
      both datasets.
    """
    text1 = pd.read_csv(
        '/content/drive/MyDrive/Colab Notebooks/Mental_Health_FAQ.csv',
        delimiter=','
    )
    text2 = pd.read_csv(
        '/content/drive/MyDrive/Colab Notebooks/medquad.csv',
        delimiter=','
    )
    # Drop unnecessary columns from both datasets
    text1 = text1.drop('Question_ID', axis=1)
    text2 = text2.drop(['source', 'focus_area'], axis=1)
    
    # Rename columns in MedQuAD dataset for consistency
    text2 = text2.rename(columns={"question": "Questions", "answer": "Answers"})
    
    # Concatenate both datasets
    text = pd.concat([text1, text2], axis=0)
    
    # Combine the Questions and Answers into a single string
    text_combined = ''.join(text['Questions'].astype(str) + text['Answers'].astype(str))
    return text_combined

def encode_text(text):
    """
    Encodes a given text into numerical representations using character-level encoding.

    This function creates two dictionaries:
    - `stoi`: Maps each character in the text to a unique index.
    - `itos`: Maps each index back to its corresponding character.
    
    The function then encodes the input text into a list of integers based on the `stoi` dictionary.

    Args:
    - text (str): The input text to be encoded.

    Returns:
    - encoded_text (list of int): A list of integers representing the encoded text.
    - stoi (dict): A dictionary that maps each character to its corresponding index.
    - itos (dict): A dictionary that maps each index back to its corresponding character.
    - vocab_size (int): The number of unique characters in the text (vocabulary size).
    """
    # Get a sorted list of unique characters from the text
    chars = sorted(list(set(text)))
    
    # Create character-to-index and index-to-character mappings
    stoi = dict(enumerate(chars))  # Refactored to use dict(enumerate(chars))
    itos = dict(enumerate(chars))
    
    # Encode the text into a list of integers using the stoi mapping
    encoded_text = [stoi[c] for c in text]
    
    return encoded_text, stoi, itos, len(chars)
