from sklearn.model_selection import train_test_split
from joblib import dump
from preprocessing import preprocess_and_save
from model import build_model


X, y = preprocess_and_save('data/Churn.csv')

X_train , X_test, y_train, y_test = train_test_split(
        X , y, test_size= 0.2, stratify= y, random_state= 42
)

model = build_model()
model.fit(X_train, y_train)

dump(model, 'models/churn_model.pkl')
print('model is trained')