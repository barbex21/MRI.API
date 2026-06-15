from fastapi import FastAPI

app = FastAPI(title="Brain MRI Tumor Inference API")

@app.get("/health")
def health():
    return {"status": "ok"}
