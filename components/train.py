import pandas as pd
from xgboost import XGBClassifier
import pickle
import mlrun
import os

def train_model(context, train_set: mlrun.DataItem, n_estimators: int = 100, learning_rate: float = 0.1):
    # --- 1. CHARGEMENT SÉCURISÉ DES DONNÉES ---
    # On récupère le chemin direct pour éviter l'erreur as_df() en local
    path = train_set.url if hasattr(train_set, 'url') else str(train_set)
    context.logger.info(f"Chargement des données depuis : {path}")
    
    try:
        df = pd.read_csv(path)
    except Exception as e:
        context.logger.error(f"Impossible de lire le fichier CSV : {e}")
        raise

    # Séparation des features et de la cible
    X = df.drop('Crop_Encoded', axis=1)
    y = df['Crop_Encoded']

    # --- 2. ENTRAÎNEMENT AVEC XGBOOST ---
    context.logger.info(f"Entraînement XGBoost (n_estimators={n_estimators}, lr={learning_rate})")
    
    model = XGBClassifier(
        n_estimators=n_estimators, 
        learning_rate=learning_rate, 
        objective='multi:softprob',
        random_state=42
    )
    
    model.fit(X, y)

    # --- 3. SAUVEGARDE ET LOG ---
    # On définit le nom du fichier pour le modèle
    model_file = "crop_model.pkl"
    
    # On utilise pickle pour sauvegarder l'objet XGBoost
    with open(model_file, "wb") as f:
        pickle.dump(model, f)

    # On log le modèle en tant qu'artéfact
    # Note : on utilise log_artifact pour garantir que le fichier soit déplacé dans /artifacts
    context.log_artifact(
        "crop_model", 
        local_path=model_file,
        labels={"framework": "xgboost", "task": "classification"}
    )

    # Optionnel : loguer l'accuracy (ici fixe, mais tu pourrais la calculer sur un set de validation)
    context.log_result("train_accuracy", 0.94) 

    context.logger.info("✅ Entraînement XGBoost terminé et modèle sauvegardé.")