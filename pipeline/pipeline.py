

import mlrun
import os

#  ÉTAPE 0 : Configuration de l'environnement local
os.environ["MLRUN_DBPATH"] = "local"
# On définit le chemin absolu pour les artefacts
os.environ["MLRUN_ARTIFACT_PATH"] = os.path.abspath("./artifacts")

# Chemins et dossiers
project_root = os.path.abspath("./")
artifact_path = os.path.join(project_root, "artifacts")

if not os.path.exists(artifact_path):
    os.makedirs(artifact_path)

# Initialisation du projet MLRun
project = mlrun.get_or_create_project("advertising-mlops", context=project_root)
project.artifact_path = artifact_path

# Définition de la fonction d'entraînement (CI)
# On lie le code source (src/train.py) à une fonction MLRun
train_fn = mlrun.code_to_function(
    name="train-function",
    filename="src/train.py",
    kind="job",
    image="mlrun/mlrun"
)

# Exécution de l'entraînement
# C'est ici que le modèle est créé et évalué
run = train_fn.run(
    handler="train_model",
    inputs={"dataset": "data/Advertising.csv"},
    artifact_path=artifact_path,
    local=True 
)

# Déploiement du modèle (CD)

serving_fn = mlrun.new_function("serving", kind="serving", image="mlrun/mlrun")

# Chemin direct vers le fichier généré par ton script train.py
model_file_path = os.path.join(artifact_path, "model.pkl")

# On configure le serveur avec le modèle et la classe de service V2
if os.path.exists(model_file_path):
    serving_fn.add_model(
        "advertising_v1", 
        model_path=model_file_path,
        class_name="mlrun.serving.V2ModelServer" 
    )
    print(f" Modèle détecté et configuré : {model_file_path}")
else:
    # Backup : utilise le dossier d'artefacts si le fichier n'est pas à la racine
    serving_fn.add_model(
        "advertising_v1", 
        model_path=artifact_path,
        class_name="mlrun.serving.V2ModelServer"
    )
    print(f"Modèle configuré via le dossier : {artifact_path}")

# Enregistrement final dans le projet
project.set_function(serving_fn)
project.save()
print("modele entrainee")