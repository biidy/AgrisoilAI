import os
import mlrun
import glob

# --- CONFIGURATION (Comme ton projet Advertising) ---
os.environ["MLRUN_DBPATH"] = "local"
# On remonte d'un niveau pour sortir du dossier pipeline/
os.environ["MLRUN_ARTIFACT_PATH"] = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "artifacts"))

import mlrun

# On prépare le dossier artifacts
artifact_path = os.environ["MLRUN_ARTIFACT_PATH"]
if not os.path.exists(artifact_path):
    os.makedirs(artifact_path)

# Initialisation du projet
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
project = mlrun.get_or_create_project("agrisoil-ai", context=project_root)

# Enregistrement des fonctions
project.set_function("components/data_prep.py", name="data-prep", kind="job", image="mlrun/mlrun")
project.set_function("components/train.py", name="train", kind="job", image="mlrun/mlrun")
project.set_function("components/evaluate.py", name="evaluate", kind="job", image="mlrun/mlrun")

def find_file(pattern):
    """Cherche un fichier dans le dossier artifacts, même s'il est dans un sous-dossier."""
    files = glob.glob(os.path.join("artifacts", "**", pattern), recursive=True)
    return files[0] if files else None

def run_mada_pipeline(source_url):
    # 1. Préparation des données
    prep = project.run_function(
        "data-prep", 
        handler="prepare_data",
        params={"source_url": source_url},
        local=True,
        artifact_path=os.path.abspath("artifacts")
    )

    # RECHERCHE DYNAMIQUE DES FICHIERS
    train_path = find_file("train_set.csv")
    test_path = find_file("test_set.csv")
    
    print(f"DEBUG: Train path trouvé -> {train_path}")

    # 2. Entraînement
    train = project.run_function(
        "train",
        handler="train_model",
        inputs={"train_set": train_path}, 
        params={"n_estimators": 100},
        local=True,
        artifact_path=os.path.abspath("artifacts")
    )

    # RECHERCHE DU MODÈLE GÉNÉRÉ
    model_path = find_file("crop_model.pkl")
    print(f"DEBUG: Model path trouvé -> {model_path}")

    # 3. Évaluation
    evaluate_run = project.run_function(
        "evaluate",
        handler="evaluate_model",
        inputs={
            "model_item": model_path, 
            "test_set": test_path
        },
        local=True,
        artifact_path=os.path.abspath("artifacts")
    )
    
    return evaluate_run

if __name__ == "__main__":
    DATA_URL = os.path.join(project_root, "data", "master_dataset_mada_30k_FINAL.csv")
    
    print("🚀 Lancement du pipeline (Ignorer les warnings de connexion API)...")
    try:
        run_mada_pipeline(source_url=DATA_URL)
        project.save()
        print("✅ Pipeline terminé avec succès !")
    except Exception as e:
        print(f"❌ Erreur critique : {e}")