import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from joblib import dump


def preprocess_and_save(path):
        df = pd.read_csv(path)
        df.drop('customerID' , axis=1, inplace=True)

        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors = 'coerce')
        df.fillna(df.median(numeric_only=True), inplace=True)

        encoders = {}

        for col in df.select_dtypes(include='object').columns:
                if col != 'Churn':
                        le = LabelEncoder()
                        df[col]  = le.fit_transform(df[col])
                        encoders[col] = le

        X = df.drop('Churn', axis=1)
        y = df['Churn']

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        dump(encoders, 'models/encoders.pkl')
        dump(scaler, 'models/scaler.pkl')

        return X_scaled, y