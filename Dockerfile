# Utilise une version "slim" pour que l'image soit plus rapide à charger
FROM python:3.9-slim

WORKDIR /app

# On installe d'abord les requirements pour profiter du cache Docker
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# On copie le reste du code
COPY . .

# On s'assure que le dossier data existe pour éviter les erreurs
RUN mkdir -p data

CMD ["python", "pipeline/pipeline.py"]