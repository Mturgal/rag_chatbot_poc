import sys
import time
import os
import openai
import datetime
import cohere
import pickle
import numpy as np
import time
from langchain_openai import ChatOpenAI
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.models import VectorParams, Distance
from qdrant_client.models import PointStruct
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_openai import OpenAIEmbeddings



# get your openai api key at https://platform.openai.com/

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
openai.api_key = OPENAI_API_KEY
# get your cohere api key at https://cohere.com/

pdf_folder_path = 'data/dataset_folder/pdf_folder'
client = QdrantClient(host="qdrant", port=6333)

client.recreate_collection(
    collection_name="novartis_collection",
    vectors_config=VectorParams(size=1536, distance=Distance.DOT),
)


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
