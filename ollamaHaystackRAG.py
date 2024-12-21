from haystack_integrations.components.generators.ollama import OllamaGenerator
from haystack import Pipeline, Document
from datasets import load_dataset
from haystack.components.retrievers.in_memory import InMemoryBM25Retriever
from haystack.components.builders.prompt_builder import PromptBuilder
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.document_stores.in_memory import InMemoryDocumentStore
import gradio as gr

template = """
Given the following information, answer the question.

Context: 
{% for document in documents %}
    {{ document.content }}
{% endfor %}

Question: {{ query }}?
"""

docstore = InMemoryDocumentStore()
docstore.write_documents([Document(content="Sita likes pet"),
                          Document(content="Sita like cute dogs"),
                          Document(content="Sita likes to pet dog")])

query="will Sita like shih Tzu?"

dataset = load_dataset("bilgeyucel/seven-wonders", split="train")
docs = [Document(content=doc["content"], meta=doc["meta"]) for doc in dataset]

#print(docs)

document_store = InMemoryDocumentStore()
document_store.write_documents(docs)

retriever = InMemoryBM25Retriever(document_store=document_store)


generator = OllamaGenerator(model="zephyr",
                            url = "http://localhost:11434",
                            generation_kwargs={
                              "num_predict": 100,
                              "temperature": 0.9,
                              })

pipe = Pipeline()
pipe.add_component("retriever", InMemoryBM25Retriever(document_store=docstore))
pipe.add_component("prompt_builder", PromptBuilder(template=template))
pipe.add_component("llm", generator)
pipe.connect("retriever", "prompt_builder.documents")
pipe.connect("prompt_builder", "llm")

def ask_question(question):
  result = pipe.run({"prompt_builder": {"query": question},
									"retriever": {"query": question}})
  return result["llm"]["replies"][0]

gr_interface = gr.Interface(
  fn=ask_question,
  inputs=gr.components.Textbox(lines=2, placeholder="Enter your question here..."),
  outputs="text"
)

gr_interface.launch()
# {'llm': {'replies': ['Based on the provided context, it seems that you enjoy
# soccer and summer. Unfortunately, there is no direct information given about 
# what else you enjoy...'],
# 'meta': [{'model': 'zephyr', ...]}}
