import mlrun
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os

def evaluate_model(context, model_item: mlrun.DataItem, test_set: mlrun.DataItem):
    # --- 1. CHARGEMENT SÉCURISÉ DU MODÈLE ---
    model_path = model_item.url if hasattr(model_item, 'url') else str(model_item)
    context.logger.info(f"Chargement du modèle depuis : {model_path}")
    
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    # --- 2. CHARGEMENT SÉCURISÉ DES DONNÉES ---
    test_path = test_set.url if hasattr(test_set, 'url') else str(test_set)
    context.logger.info(f"Chargement des données de test depuis : {test_path}")
    
    df_test = pd.read_csv(test_path)
    X_test = df_test.drop('Crop_Encoded', axis=1)
    y_test = df_test['Crop_Encoded']
    
    # --- 3. PRÉDICTIONS ---
    y_pred = model.predict(X_test)
    
    # --- 4. CALCUL DES MÉTRIQUES ---
    accuracy = accuracy_score(y_test, y_pred)
    
    # Calcul de Precision, Recall et F1-score (moyenne pondérée pour le multi-classe)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')
    
    # Rapport détaillé par culture (pour le log dataset)
    report = classification_report(y_test, y_pred, output_dict=True)
    
    # --- 5. GÉNÉRATION DE LA MATRICE DE CONFUSION ---
    plt.figure(figsize=(12, 8))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f"Matrice de Confusion - Accuracy: {accuracy:.4f}")
    plt.xlabel('Prédictions')
    plt.ylabel('Réel')
    
    plot_path = "confusion_matrix.png"
    plt.savefig(plot_path)
    plt.close() # Ferme la figure pour libérer la mémoire
    
    # --- 6. LOG DES RÉSULTATS DANS MLRUN ---
    # Log du rapport complet en CSV
    context.log_dataset("evaluation_report", df=pd.DataFrame(report).transpose(), index=True)
    
    # Log de l'image (Artefact visuel pour le jury)
    context.log_artifact("confusion_matrix_plot", local_path=plot_path)
    
    # Log des métriques clés (Variables scalaires)
    context.log_result("accuracy", float(accuracy))
    context.log_result("precision", float(precision))
    context.log_result("recall", float(recall))
    context.log_result("f1_score", float(f1))
    
    # Affichage propre dans les logs GitHub Actions
    print("\n" + "="*40)
    print("🚀 RÉSULTATS DE L'ÉVALUATION")
    print("="*40)
    print(f"Accuracy  : {accuracy:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1-Score  : {f1:.4f}")
    print("="*40)
    
    context.logger.info(f"✅ Évaluation terminée. Accuracy : {accuracy:.4f}")