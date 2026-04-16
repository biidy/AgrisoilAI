import mlrun
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

def evaluate_model(context, model_item: mlrun.DataItem, test_set: mlrun.DataItem):
    # 1. Charger le modèle et les données
    model = model_item.as_model()
    df_test = test_set.as_df()
    
    X_test = df_test.drop('Crop_Encoded', axis=1)
    y_test = df_test['Crop_Encoded']
    
    # 2. Prédictions
    y_pred = model.predict(X_test)
    
    # 3. Calcul des métriques
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    
    # 4. Génération de la Matrice de Confusion (Visuelle)
    plt.figure(figsize=(12, 8))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f"Matrice de Confusion - Accuracy: {accuracy:.2f}")
    
    # 5. Log des résultats dans MLRun
    context.log_dataset("evaluation_results", df=pd.DataFrame(report).transpose())
    context.log_plot("confusion_matrix", plt.gcf())
    context.log_result("accuracy", accuracy)
    
    context.logger.info(f"Évaluation terminée. Accuracy : {accuracy:.4f}")