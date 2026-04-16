import os

# --- ÉTAPE 0 : CONFIGURATION RÉSEAU (DOIT ÊTRE AU DÉBUT) ---
# Force MLRun à travailler en local
os.environ["MLRUN_DBPATH"] = "local"
# On remonte d'un niveau (../) pour que les artefacts soient à la racine du projet et non dans /pipeline
os.environ["MLRUN_ARTIFACT_PATH"] = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "artifacts"))

import mlrun

# --- ÉTAPE 1 : INITIALISATION DU PROJET ---
# Le context doit être la racine du projet (le parent du dossier pipeline)
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
artifact_path = os.environ["MLRUN_ARTIFACT_PATH"]

if not os.path.exists(artifact_path):
    os.makedirs(artifact_path)

project = mlrun.get_or_create_project("agrisoil-ai", context=project_root)

# --- ÉTAPE 2 : ENREGISTREMENT DES COMPOSANTS ---
# On utilise des chemins relatifs à partir de la racine du projet (project_root)
project.set_function("components/data_prep.py", name="data-prep", kind="job", image="mlrun/mlrun")
project.set_function("components/train.py", name="train", kind="job", image="mlrun/mlrun")
project.set_function("components/evaluate.py", name="evaluate", kind="job", image="mlrun/mlrun")

# --- ÉTAPE 3 : DÉFINITION DU WORKFLOW ---
def run_mada_pipeline(source_url):
    """Orchestre les 3 étapes du projet."""
    
    # 1. Préparation des données
    prep = project.run_function(
        "data-prep", 
        handler="prepare_data",
        params={"source_url": source_url},
        local=True
    )

    # 2. Entraînement
    train = project.run_function(
        "train",
        handler="train_model",
        inputs={"train_set": prep.outputs['train_set']},
        params={"n_estimators": 100},
        local=True
    )

    # 3. Évaluation
    evaluate_run = project.run_function(
        "evaluate",
        handler="evaluate_model",
        inputs={
            "model_item": train.outputs['crop_model'], 
            "test_set": prep.outputs['test_set']
        },
        local=True
    )
    
    return evaluate_run

# --- ÉTAPE 4 : POINT D'ENTRÉE ---
if __name__ == "__main__":
    # Chemin vers le dataset à partir de la racine
    DATA_URL = os.path.join(project_root, "data", "master_dataset_mada_30k_FINAL.csv")
    
    print(f"🚀 Démarrage du pipeline depuis : {os.path.dirname(__file__)}")
    
    try:
        run_mada_pipeline(source_url=DATA_URL)
        project.save()
        print("✅ Pipeline terminé avec succès !")
        print(f"📁 Artefacts disponibles dans : {artifact_path}")
    except Exception as e:
        print(f"❌ Erreur : {e}")