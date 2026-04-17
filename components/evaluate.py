import mlrun
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os

def evaluate_model(context, model_item: mlrun.DataItem, test_set: mlrun.DataItem):
    # --- 1. CHARGEMENT SÉCURISÉ DU MODÈLE ---
    # On récupère le chemin du fichier (soit depuis l'objet MLRun, soit le string direct)
    model_path = model_item.url if hasattr(model_item, 'url') else str(model_item)
    context.logger.info(f"Chargement du modèle depuis : {model_path}")
    
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    # --- 2. CHARGEMENT SÉCURISÉ DES DONNÉES ---
    test_path = test_set.url if hasattr(test_set, 'url') else str(test_set)
    context.logger.info(f"Chargement des données de test depuis : {test_path}")
    
    df_test = pd.read_csv(test_path)
    
    # Préparation des features et de la cible
    X_test = df_test.drop('Crop_Encoded', axis=1)
    y_test = df_test['Crop_Encoded']
    
    # --- 3. PRÉDICTIONS ---
    y_pred = model.predict(X_test)
    
    # --- 4. CALCUL DES MÉTRIQUES ---
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    
    # --- 5. GÉNÉRATION DE LA MATRICE DE CONFUSION ---
    plt.figure(figsize=(12, 8))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f"Matrice de Confusion - Accuracy: {accuracy:.2f}")
    
    # Sauvegarde locale du plot avant de l'envoyer à MLRun (plus stable en local)
    plot_path = "confusion_matrix.png"
    plt.savefig(plot_path)
    
    # --- 6. LOG DES RÉSULTATS DANS MLRUN ---
    # Log du dataset de rapport
    context.log_dataset("evaluation_results", df=pd.DataFrame(report).transpose(), index=True)
    
    # Log de l'image de la matrice
    context.log_artifact("confusion_matrix_plot", local_path=plot_path)
    
    # Log de la valeur simple
    context.log_result("accuracy", float(accuracy))
    
    context.logger.info(f"✅ Évaluation terminée avec succès. Accuracy : {accuracy:.4f}")