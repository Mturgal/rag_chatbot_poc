import os
import datetime
from flask import Flask, flash, request, redirect, url_for, jsonify, render_template, session
import json
import tiktoken
import re
import cohere
import openai
import pickle
import numpy as np
import time
from langchain_openai import ChatOpenAI
import config
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFDirectoryLoader
from qdrant_client import QdrantClient
from qdrant_client import models

app = Flask(__name__)
app.config['SECRET_KEY'] = 'this_is_bad_secret_key'

OPENAI_API_KEY = config.OPENAI_API_KEY
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
openai.api_key = OPENAI_API_KEY

limit_input_tokens=4096
pdf_folder_path = 'data/dataset_folder/pdf_folder'
prompt_template_path = 'data/prompt_template/prompt_template.txt'
num_chunks_of_text = 8 # number of chunks of text to search via embeddings similarity
client = QdrantClient(url="http://localhost:6333")


def read_template():
    # read file
    with open(prompt_template_path, 'r', encoding="utf-8") as f:
        prompt_template = f.read()
    return prompt_template

prompt_template = read_template()

def get_context_embeddings_openai(doc_texts):
    res = openai.embeddings.create(input=doc_texts, model="text-embedding-ada-002")
    doc_embeds = [r.embedding for r in res.data]
    return np.asarray(doc_embeds)


def generate_full_llm_query(client, query, prompt_template, limit_input_tokens=4096):

    query_emb = get_context_embeddings_openai(query)
    top_results = client.search(
    collection_name="novartis_collection",
    search_params=models.SearchParams(hnsw_ef=128, exact=False),
    query_vector=query_emb[0],
    limit=num_chunks_of_text,
)
    llm_full_query = ''

    # More complex logic is implemented to ensure the full llm query does not exceeds input token limit
    # i.e. finall llm query is cooked with size < limit_input_tokens
    correction_num_of_tokens = 500 # additionally decrease num of tokens by this number
    # Truncating chunks to the size of limit_input_tokens
    num_of_chunks = len(top_results)
    for _ in range(num_of_chunks):
        print('top results len:', len(top_results))
        context_chunks_as_str = '\n###\n'.join([str(elem.payload["page_content"]) for elem in top_results])
        llm_full_query = prompt_template.format(context=context_chunks_as_str, question=query)
        encoding = tiktoken.get_encoding("cl100k_base")
        num_tokens_template = len(encoding.encode(prompt_template))
        num_tokens = len(encoding.encode(llm_full_query))
        print('num_tokens', num_tokens)

        if num_tokens <= limit_input_tokens:
          return llm_full_query, num_tokens
        elif len(top_results) == 1 and num_tokens > limit_input_tokens - correction_num_of_tokens:
          chunk_appendix = '\nDetails in the link:'
          # extracted_link = re.search(r'https://.+', context_chunks[0][-100:]).group(0)
          extracted_link = top_results[0].metadata["source"]
          chunk_appendix = chunk_appendix + ' ' + extracted_link
          num_of_chars_to_cut = num_tokens - limit_input_tokens + num_tokens_template + correction_num_of_tokens
          top_results[0].payload["page_content"] = top_results[0].payload["page_content"][:-num_of_chars_to_cut]
          top_results[0].payload["page_content"] = top_results[0].payload["page_content"] + chunk_appendix
          print(top_results[0].payload["page_content"])
          context_chunks_as_str = '\n###\n'.join([str(elem.payload["page_content"]) for elem in top_results])
          llm_full_query = prompt_template.format(context=context_chunks_as_str, question=query)
          num_tokens = len(encoding.encode(llm_full_query))
          return llm_full_query, num_tokens
        elif num_tokens > limit_input_tokens - correction_num_of_tokens:
          top_results = top_results[:-1]

    return llm_full_query, context_chunks


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/send_message', methods=['POST'])
def send_message():
    data = request.get_json()
    if isinstance(data, str):
        data = json.loads(data)
    
    query = data['message']

    # Process user message here if needed
    now = datetime.datetime.now()
    llm_full_query, context_chunks = generate_full_llm_query(client, query, prompt_template, limit_input_tokens=limit_input_tokens)
    # print('current full query:', llm_full_query)
    llm_answer = llm.invoke(llm_full_query)
    #llm_answer = llm_full_query  # this is a stub if llm initialization disabled
    llm_answer = llm_answer.content.strip()

    after = datetime.datetime.now()
    delta_time = after - now
    delta_time = round(delta_time.total_seconds())
    llm_answer = llm_answer + '\n\n' + '> response generation time: ' + str(delta_time) + ' seconds.'
    print(llm_answer)
    response = {'message': llm_answer}
    return jsonify(response)


if __name__ == "__main__":
    llm = ChatOpenAI(model_name='gpt-3.5-turbo')
    app.run(host='0.0.0.0', port=80)
