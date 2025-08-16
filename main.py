from fastapi import FastAPI
app = FastAPI()

@app.get("/")
async def root():
    return {"message": "API is working"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
