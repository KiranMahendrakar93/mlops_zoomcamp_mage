import pandas as pd
from sklearn.model_selection import train_test_split
from typing import Tuple

def split(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    
    X = df.drop(columns=['GradeClass'])
    y = df['GradeClass']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    return X_train, X_test, y_train, y_test