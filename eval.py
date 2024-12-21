import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report
import json

def evaluate_model():
 
    data = pd.read_csv('data/iris.csv')
    X = data.drop(columns='target')
    y = data['target']

    model = joblib.load('models/iris_model.pkl')
    predictions = model.predict(X)
    
  
    report = classification_report(y, predictions, target_names=['setosa', 'versicolor', 'virginica'], output_dict=True)


    with open('metrics/report.txt', 'w') as f:
        f.write(classification_report(y, predictions, target_names=['setosa', 'versicolor', 'virginica']))
    

    with open('metrics/report.json', 'w') as json_file:
        json.dump(report, json_file)


    metrics = report['setosa'], report['versicolor'], report['virginica']

    df = pd.DataFrame({
        'Class': ['setosa', 'versicolor', 'virginica'],
        'Precision': [metrics[0]['precision'], metrics[1]['precision'], metrics[2]['precision']],
        'Recall': [metrics[0]['recall'], metrics[1]['recall'], metrics[2]['recall']],
        'F1-Score': [metrics[0]['f1-score'], metrics[1]['f1-score'], metrics[2]['f1-score']]
    })

    df.set_index('Class').plot(kind='bar', figsize=(10, 6))
    plt.title('Classification Metrics for Iris Dataset')
    plt.ylabel('Score')
    plt.xlabel('Class')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('metrics/classification_report.png') 
    plt.show()

if __name__ == "__main__":
    evaluate_model()
