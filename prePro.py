import pandas as pd
def load_and_process_data():
    df = pd.read_csv("housing.csv")
    df.replace('yes', 1, inplace=True)
    df.replace('no', 0, inplace=True)
    df.replace('furnished', 1, inplace=True)
    df.replace('unfurnished', 0, inplace=True)
    df.replace('semi-furnished', -1, inplace=True)

    # Feature engineering

    return df
processed_data = load_and_process_data()