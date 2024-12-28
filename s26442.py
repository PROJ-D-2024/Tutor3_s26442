import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


def data_preprocessing(df, test_size=0.2):
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df['Timestamp'] = df['Timestamp'].apply(lambda x: x.value)
    df['Timestamp'] = (df['Timestamp'] - df['Timestamp'].min()) / (df['Timestamp'].max() - df['Timestamp'].min())

    cat_cols = df.select_dtypes(include='object').columns.to_list()

    le = LabelEncoder()
    for col in cat_cols:
        df[col] = le.fit_transform(df[col])
    
    scaler = MinMaxScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

    y = df_scaled['Is Laundering']
    X = df_scaled.drop('Is Laundering', axis=1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=y, random_state=42)
    return X_train, X_test, y_train, y_test

df = pd.read_csv('/Users/andrzej/python_programs/CODE/data/ibm-transactions-for-anti-money-laundering-aml/HI-Small_Trans.csv')
X_train, X_test, y_train, y_test = data_preprocessing(df)