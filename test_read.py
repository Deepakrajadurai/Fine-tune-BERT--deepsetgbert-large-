import pandas as pd

df = pd.read_csv("Data/train.csv", nrows=1)
text = df["text"].iloc[0]
print("No encoding specified, repr:")
print(repr(text))
print("Chars in 'gültigen' or whatever it is:")
print([ord(c) for c in text])

# Let's try reading with cp1252
try:
    df_cp = pd.read_csv("Data/train.csv", nrows=1, encoding="cp1252")
    text_cp = df_cp["text"].iloc[0]
    print("\nCP1252, repr:")
    print(repr(text_cp))
except Exception as e:
    print("CP1252 error:", e)

# Let's try reading with utf-8
try:
    df_utf8 = pd.read_csv("Data/train.csv", nrows=1, encoding="utf-8")
    text_utf8 = df_utf8["text"].iloc[0]
    print("\nUTF-8, repr:")
    print(repr(text_utf8))
except Exception as e:
    print("UTF-8 error:", e)
