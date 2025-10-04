# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional, List
import pandas as pd
import io
import json
from agents import DataAnalysisAgents, MODEL_CONFIGS

app = FastAPI(title="Data Analysis Agent API")

# CORS middleware for Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001"],  # Next.js ports
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory storage for uploaded datasets (use Redis/DB for production)
datasets = {}

class QueryRequest(BaseModel):
    dataset_id: str
    query: str
    model_key: str = "llama-3-1-nemotron-ultra-v1"
    chat_context: Optional[str] = None

class ModelInfo(BaseModel):
    key: str
    name: str
    url: str

@app.get("/")
async def root():
    return {"message": "Data Analysis Agent API", "version": "1.0.0"}

@app.get("/api/models", response_model=List[ModelInfo])
async def get_models():
    """Get available models."""
    return [
        ModelInfo(
            key=key,
            name=config.MODEL_PRINT_NAME,
            url=config.MODEL_URL
        )
        for key, config in MODEL_CONFIGS.items()
    ]

@app.post("/api/upload")
async def upload_dataset(file: UploadFile = File(...), model_key: str = "llama-3-1-nemotron-ultra-v1"):
    """Upload CSV file and generate EDA analysis."""
    try:
        # Read CSV file
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))

        # Generate unique dataset ID
        dataset_id = f"dataset_{len(datasets)}"
        datasets[dataset_id] = df

        # Initialize agents with selected model
        agents = DataAnalysisAgents(model_key=model_key)

        # Generate comprehensive EDA
        eda_results = agents.comprehensive_eda_agent(df)

        # Generate insights
        insights = agents.data_insight_agent(df)

        # Get dataset preview
        preview = df.head(10).to_dict('records')

        return {
            "dataset_id": dataset_id,
            "filename": file.filename,
            "preview": preview,
            "columns": list(df.columns),
            "shape": df.shape,
            "eda_results": eda_results,
            "insights": insights
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/api/query")
async def query_dataset(request: QueryRequest):
    """Process a natural language query on the dataset."""
    try:
        # Get dataset
        if request.dataset_id not in datasets:
            raise HTTPException(status_code=404, detail="Dataset not found")

        df = datasets[request.dataset_id]

        # Initialize agents with selected model
        agents = DataAnalysisAgents(model_key=request.model_key)

        # Generate code
        code, should_plot = agents.code_generation_agent(
            request.query,
            df,
            request.chat_context
        )

        # Execute code
        result = agents.execution_agent(code, df, should_plot)

        # Generate reasoning
        thinking, explanation = agents.reasoning_agent(request.query, result)

        return {
            "code": code,
            "result": result,
            "thinking": thinking,
            "explanation": explanation,
            "should_plot": should_plot
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/chat")
async def chat_stream(request: QueryRequest):
    """Stream chat responses for real-time interaction."""
    try:
        if request.dataset_id not in datasets:
            raise HTTPException(status_code=404, detail="Dataset not found")

        df = datasets[request.dataset_id]
        agents = DataAnalysisAgents(model_key=request.model_key)

        async def generate():
            # Generate code
            code, should_plot = agents.code_generation_agent(
                request.query,
                df,
                request.chat_context
            )

            yield json.dumps({"type": "code", "data": code}) + "\n"

            # Execute code
            result = agents.execution_agent(code, df, should_plot)
            yield json.dumps({"type": "result", "data": result}) + "\n"

            # Generate reasoning
            thinking, explanation = agents.reasoning_agent(request.query, result)
            yield json.dumps({"type": "thinking", "data": thinking}) + "\n"
            yield json.dumps({"type": "explanation", "data": explanation}) + "\n"

        return StreamingResponse(generate(), media_type="application/x-ndjson")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/dataset/{dataset_id}")
async def delete_dataset(dataset_id: str):
    """Delete a dataset from memory."""
    if dataset_id in datasets:
        del datasets[dataset_id]
        return {"message": "Dataset deleted successfully"}
    raise HTTPException(status_code=404, detail="Dataset not found")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
