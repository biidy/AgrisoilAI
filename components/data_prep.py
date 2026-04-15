import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import mlrun

def prepare_data(context, source_url: str):
    df = pd.read_csv(source_url)
    
    # Encodage
    le_region = LabelEncoder()
    le_crop = LabelEncoder()
    
    df['Region_Encoded'] = le_region.fit_transform(df['Region'])
    df['Crop_Encoded'] = le_crop.fit_transform(df['Crop'])
    
    # On garde les noms des colonnes pour le futur
    features = ['Altitude', 'Mois', 'Saison', 'SOC', 'Clay', 'Nitrogen', 
                'CEC', 'Temp', 'Hum', 'pH', 'Rain', 'Lux', 'CE', 'Region_Encoded']
    
    X = df[features]
    y = df['Crop_Encoded']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Log des artefacts (très important pour le MLOps)
    context.log_dataset("train_set", df=pd.concat([X_train, y_train], axis=1), format="csv")
    context.log_dataset("test_set", df=pd.concat([X_test, y_test], axis=1), format="csv")
    context.log_artifact("region_encoder", body=le_region, local_path="le_region.pkl")
    context.log_artifact("crop_encoder", body=le_crop, local_path="le_crop.pkl")
    
    context.logger.info("Préparation des données terminée.")