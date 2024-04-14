import pandas as pd
import numpy as np

def trim_quantiles(df, column, quantiles=100):
    quantile_values = df[column].quantile(q=np.linspace(0, 1, quantiles))
    # Trim based on quantile
    df[f'{column}_trim'] = df[column].where(df[column] < quantile_values.iloc[-2])
    return df