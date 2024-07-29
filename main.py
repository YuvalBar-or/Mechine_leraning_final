import pandas as pd
from collections import Counter


def count_labels(csv_file_path):
    df = pd.read_csv(csv_file_path)
    if 'label' not in df.columns:
        raise ValueError("The CSV file does not contain a 'label' column.")
    label_counts = Counter(df['label'])

    return label_counts


csv_file_path = 'data.csv'
label_counts = count_labels(csv_file_path)

for label, count in label_counts.items():
    print(f"{label}: {count}")




