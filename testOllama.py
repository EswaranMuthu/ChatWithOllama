import ollama
desiredModel= 'llama3.2' 
questionToAsk='What is quantum computing'
response = ollama.chat(model=desiredModel, messages=[
    {
    'role': 'user',
    'content': questionToAsk,
    },
])
OllamaResponse=response['message']['content']
with open("outputOllama.txt", "w", encoding="utf-8") as text_file:
    text_file.write(OllamaResponse)
