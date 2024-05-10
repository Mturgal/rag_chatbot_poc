import sys
import config
import time
import os
import openai
import datetime
#from flask import Flask, flash, request, redirect, url_for, jsonify, render_template, session
import cohere
import pickle
import numpy as np
import time
from langchain_openai import ChatOpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from qdrant_client.http import models
from qdrant_client.models import PointStruct
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_openai import OpenAIEmbeddings



# get your openai api key at https://platform.openai.com/
OPENAI_API_KEY = config.OPENAI_API_KEY
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
openai.api_key = OPENAI_API_KEY
# get your cohere api key at https://cohere.com/
COHERE_KEY = config.COHERE_KEY
co = cohere.Client(COHERE_KEY)

file_path = 'data/dataset_folder/dataset_novartis'
pdf_folder_path = 'data/dataset_folder/pdf_folder'
num_chunks_of_text = 8 # number of chunks of text to search via embeddings similarity
client = QdrantClient(host="localhost", port=6333)



loader = PyPDFDirectoryLoader(pdf_folder_path)
docs = loader.load()

payload = []
for doc in docs:
    payload.append({"page_content": doc.page_content, "source": doc.metadata["source"], "page": doc.metadata["page"]})
indexes = range(len(payload))
doc_texts = [doc.page_content if doc.page_content != "" else "nothing" for doc in docs ]

def get_context_embeddings_openai(doc_texts):
    res = openai.embeddings.create(input=doc_texts, model="text-embedding-ada-002")
    doc_embeds = [r.embedding for r in res.data]
    return np.asarray(doc_embeds)

context_emb = get_context_embeddings_openai(doc_texts)

client.upload_collection(
    collection_name="novartis_collection",
    ids=indexes,
    payload=payload,
    vectors=context_emb,
    max_retries=5,
    )
