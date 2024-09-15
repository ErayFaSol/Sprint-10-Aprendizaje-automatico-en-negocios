from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from math import sqrt


def process_data(data):
     # División de los datos
    X = data[['f0', 'f1', 'f2']]
    y = data['product']
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.25, random_state=123)
    
    # Entrenamiento del modelo
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Predicciones y evaluación
    predictions = model.predict(X_valid)
    rmse = sqrt(mean_squared_error(y_valid, predictions))
    mean_product = predictions.mean()
    
    return rmse, mean_product, predictions
