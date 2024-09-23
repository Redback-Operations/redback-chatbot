import pandas as pd


def load_and_preprocess_data():
    text1 = pd.read_csv(
        '/content/drive/MyDrive/Colab Notebooks/Mental_Health_FAQ.csv',
        delimiter=','
    )
    text2 = pd.read_csv(
        '/content/drive/MyDrive/Colab Notebooks/medquad.csv',
        delimiter=','
    )
    text1 = text1.drop('Question_ID', axis=1)
    text2 = text2.drop(['source', 'focus_area'], axis=1)
    text2 = text2.rename(columns={"question": "Questions", "answer": "Answers"})
    text = pd.concat([text1, text2], axis=0)
    text_combined = ''.join(text['Questions'].astype(str) +
                            text['Answers'].astype(str))
    return text_combined


def encode_text(text):
    chars = sorted(list(set(text)))
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    return [stoi[c] for c in text], stoi, itos, len(chars)
