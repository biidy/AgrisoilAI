import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import mlrun
import pickle # Indispensable pour transformer les objets en fichiers
import os

def prepare_data(context, source_url: str):
    context.logger.info(f"Chargement des données depuis : {source_url}")
    df = pd.read_csv(source_url)
    
    # --- ENCODAGE ---
    le_region = LabelEncoder()
    le_crop = LabelEncoder()
    
    df['Region_Encoded'] = le_region.fit_transform(df['Region'])
    df['Crop_Encoded'] = le_crop.fit_transform(df['Crop'])
    
    # --- SELECTION DES FEATURES ---
    features = ['Altitude', 'Mois', 'Saison', 'SOC', 'Clay', 'Nitrogen', 
                'CEC', 'Temp', 'Hum', 'pH', 'Rain', 'Lux', 'CE', 'Region_Encoded']
    
    X = df[features]
    y = df['Crop_Encoded']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # --- LOG DES JEUX DE DONNÉES ---
    context.log_dataset("train_set", df=pd.concat([X_train, y_train], axis=1), format="csv", index=False)
    context.log_dataset("test_set", df=pd.concat([X_test, y_test], axis=1), format="csv", index=False)
    
    # --- LOG DES ENCODEURS (LA CORRECTION EST ICI) ---
    # On sauvegarde les encodeurs dans des fichiers locaux temporaires
    region_path = "le_region.pkl"
    crop_path = "le_crop.pkl"
    
    with open(region_path, "wb") as f:
        pickle.dump(le_region, f)
        
    with open(crop_path, "wb") as f:
        pickle.dump(le_crop, f)
    
    # On indique à MLRun de récupérer ces fichiers
    context.log_artifact("region_encoder", local_path=region_path)
    context.log_artifact("crop_encoder", local_path=crop_path)
    
    context.logger.info("✅ Préparation des données et sauvegarde des encodeurs terminées.")