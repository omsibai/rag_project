from fastapi import FastAPI
from pydantic import BaseModel
from rag_local import get_llm_resp
import json


def preprocessing(output):
    res = output.split('```')[1].replace('json', '', 1)
    return json.loads(res)
    
app = FastAPI()


@app.get("/")
async def get_resp(q:str):
    return preprocessing(get_llm_resp(q))