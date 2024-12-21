import pandas as pd
import joblib
import yaml
import json
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def train_model():
    # Загрузка параметров из params.yaml
    with open("params.yaml", "r") as file:
        params = yaml.safe_load(file)

    # Загрузка данных
    data = pd.read_csv('data/iris.csv')
    X = data.drop(columns='target')
    y = data['target']

    # Разделение данных на тренировочную и тестовую выборки
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Настройка модели с параметрами из params.yaml
    model = RandomForestClassifier(
        n_estimators=params['random_forest']['n_estimators'],
        max_depth=params['random_forest']['max_depth'],
        random_state=42
    )

    # Обучение модели
    model.fit(X_train, y_train)

    # Прогнозирование и оценка модели
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    print(f"Model Accuracy: {accuracy}")

    # Сохранение метрик в JSON
    metrics = {"accuracy": accuracy}
    with open("results/train/metrics.json", "w") as metrics_file:
        json.dump(metrics, metrics_file)

    # Сохранение модели
    joblib.dump(model, 'models/iris_model.pkl')

if __name__ == "__main__":
    train_model()
