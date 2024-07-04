import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from typing import Tuple, Dict


def prep(df: pd.DataFrame) -> pd.DataFrame:
    
    #  numeric and categoric columns
    columns = list(df.columns)

    categoric_columns = []
    numeric_columns = []

    for i in columns:
        if len(df[i].unique()) > 5:
            numeric_columns.append(i)
        else:
            categoric_columns.append(i)

    label_encoders = {}
    for column in df[categoric_columns]:  
        label_encoder = LabelEncoder()
        df[column] = label_encoder.fit_transform(df[column])
        label_encoders[column] = label_encoder
    
    std_scalers = {}
    for column in df[numeric_columns]:
        scaler = StandardScaler()
        df[column] = scaler.fit_transform(df[[column]])
        std_scalers[column] = scaler
    
    print(std_scalers[0])

    return df