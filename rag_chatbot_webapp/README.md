# RAG Chatbot with Public Novartis Data

This is a simple chatbot based on crawled Novartis data with apify.com.
To run the chatbot, 
1- Add a config file under 'rag_chatbot_poc' folder.
please insert in the config.py the api keys for openai and cohere

```
OPENAI_API_KEY=<api_key>
COHERE_KEY =<api_key>
```

2-Install the dependencies:
First build a virtual environment:
```
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

3-Run the app:
```
python webapp_novartis.py
```

4-If you want to serve the app on web for testing, the simplest way is to do
this with ngrok, for setting up please
[visit](https://ngrok.com/docs/guides/getting-started/)  

```
ngrok http 80
```





This chatbot is based on this blog
[post](https://medium.com/@vovakuzmenkov/building-a-fullstack-rag-solution-with-private-llm-a-step-by-step-guide-48a0a4467efc#0a34) 

