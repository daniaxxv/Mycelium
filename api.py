from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
import pandas as pd
from io import StringIO
from app import UserClusteringAgent  # Заміни на ім'я файлу з класом, e.g., app.py

app = FastAPI(
    title="AI User Clustering API",
    description="API for energy-efficient user clustering",
    version="1.0"
)

# Модель для вхідних даних (JSON body)
class UserData(BaseModel):
    users: list[dict]  # e.g., [{"user_id": "1", "age": 25, ...}]

@app.get("/")
def root():
    return {"message": "Welcome to AI Clustering API. Docs at /docs"}

@app.post("/cluster")
def cluster_users(data: UserData, n_clusters: int = 5):
    # Конвертуй JSON в DataFrame
    df = pd.DataFrame(data.users)
    
    agent = UserClusteringAgent()  # Ініціалізуй твій агент
    results, metrics = agent.run_clustering(df, n_clusters)
    
    # Поверни JSON з результатами
    return {
        "clusters": results.to_dict(orient="records"),
        "metrics": metrics,
        "energy_saved": metrics.get("energy_saved", 0)  # Адаптуй під твої метрики
    }

@app.post("/cluster-file")
async def cluster_from_file(file: UploadFile = File(...), n_clusters: int = 5):
    # Читання CSV файлу
    content = await file.read()
    df = pd.read_csv(StringIO(content.decode("utf-8")))
    
    agent = UserClusteringAgent()
    results, metrics = agent.run_clustering(df, n_clusters)
    
    return {
        "clusters": results.to_dict(orient="records"),
        "metrics": metrics
    }

# Запуск: uvicorn api:app --reload