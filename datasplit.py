import pandas as pd
from sklearn.model_selection import train_test_split


def split_data(csv_file_path, train_file_path, test_file_path, test_size=0.2, random_state=42):
    data = pd.read_csv(csv_file_path)
    
    #documentation on https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
    # random_state controls the shuffling applied to the data before applying the split. Pass an int for reproducible output across multiple function calls.
    train_data, test_data = train_test_split(data, test_size=test_size, random_state=random_state)
    
    train_data.to_csv(train_file_path, index=False)
    test_data.to_csv(test_file_path, index=False)
