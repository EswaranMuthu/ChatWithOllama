from fastapi import FastAPI
import ollama
desiredModel= 'llama3.2' 
questionToAsk='What is quantum computing'

app = FastAPI()

@app.get("/hellollama/{query}")
def helloLLAMA(query : str) -> str:
    response = ollama.chat(model=desiredModel, messages=[
    {
    'role': 'user',
    'content': query,
    },
    ])
    OllamaResponse=response['message']['content']
    return OllamaResponse