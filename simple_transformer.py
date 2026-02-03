pip install transformers torch

qa_pipeline = pipeline("question-answering", model="distilbert-base-uncased-distilled-squad") 

context = "Hugging Face is a technology company that provides open-source NLP libraries ..." 
question = "What does Hugging Face provide?" 
answer = qa_pipeline(question=question, context=context) 
print(f"Question: {question}") 
print(f"Answer: {answer['answer']}"
