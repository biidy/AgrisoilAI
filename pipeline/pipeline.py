import os

# --- CETTE PARTIE DOIT ÊTRE LA TOUTE PREMIÈRE ---
os.environ["MLRUN_DBPATH"] = "local"
os.environ["MLRUN_ARTIFACT_PATH"] = "./artifacts"
# -----------------------------------------------


import mlrun


project_root = os.path.abspath("./")
if not os.path.exists("./artifacts"):
    os.makedirs("./artifacts")

# --- INITIALISATION DU PROJET ---
# On remplace l'import complexe de KFP par la gestion native du projet
project = mlrun.get_or_create_project("agrisoil-ai", context=project_root)

# On enregistre les fonctions une seule fois
project.set_function("components/data_prep.py", name="data-prep", kind="job", image="mlrun/mlrun")
project.set_function("components/train.py", name="train", kind="job", image="mlrun/mlrun")
project.set_function("components/evaluate.py", name="evaluate", kind="job", image="mlrun/mlrun")

# --- LE WORKFLOW (Version Simplifiée) ---
# Au lieu d'utiliser @dsl.pipeline qui est sensible aux versions de KFP,
# on utilise une fonction Python standard que MLRun va orchestrer.

def run_mada_pipeline(source_url):
    # Étape 1 : Préparation
    prep = project.run_function(
        "data-prep", 
        handler="prepare_data",
        params={"source_url": source_url},
        local=True
    )

    # Étape 2 : Entraînement
    train = project.run_function(
        "train",
        handler="train_model",
        inputs={"train_set": prep.outputs['train_set']},
        params={"n_estimators": 100},
        local=True
    )

    # Étape 3 : Évaluation
    evaluate_run = project.run_function(
    "evaluate",
    handler="evaluate_model", # Nom de la fonction dans evaluate.py
    inputs={
        "model_item": train_run.outputs['crop_model'], # Assure-toi que c'est 'crop_model'
        "test_set": prep_run.outputs['test_set']
    },
    local=True
    )
    return evaluate

# --- EXÉCUTION ---
if __name__ == "__main__":
    DATA_URL = "./data/master_dataset_mada_30k_FINAL.csv"
    
    # Lancement du workflow
    run_mada_pipeline(source_url=DATA_URL)
    
    # Sauvegarde finale pour GitHub
    project.save()
    print("Pipeline exécuté avec succès et projet sauvegardé.")