import pandas as pd


def clean(df: pd.DataFrame) -> pd.DataFrame:
    
    #  numeric and categoric columns
    columns = list(df.columns)

    categoric_columns = []
    numeric_columns = []

    for i in columns:
        if len(df[i].unique()) > 5:
            numeric_columns.append(i)
        else:
            categoric_columns.append(i)

        
    # Assuming the first column is an ID or non-numeric feature
    numeric_columns = numeric_columns[1:]

    # Convert numeric columns to float64
    df[numeric_columns] = df[numeric_columns].astype('float64')

    df = df.drop(columns=['GPA', 'StudentID', 'Age'])

    return df
