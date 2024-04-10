# ThesisBot 🤖
ThesisBot is your intelligent companion for navigating through intricate thesis documents. Powered by LLM, this chatbot offers an interactive platform for querying and exploring thesis content. Whether you need summaries, clarifications, or deeper insights, ThesisBot is here to assist you. With its user-friendly Streamlit interface, ThesisBot provides a conversational experience that simplifies the process of engaging with academic research. 

# How to run?
### STEPS:

Clone the repository

```bash
Project repo: https://github.com/arun-lal/thesis-bot.git
```

### STEP 01- Create a conda environment after opening the repository

```bash
python create -m venv .venv 
```

```bash
python activate .venv
```

### STEP 02- install the requirements
```bash
pip install -r requirements.txt
```


### Create a `.env` file in the root directory and add your Pinecone credentials as follows:

```ini
PINECONE_API_KEY = "xxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
PINECONE_API_ENV = "xxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
```


### Download the quantize model from the link provided in model folder & keep the model in the model directory:

```ini
## Download the Llama 2 Model:

llama-2-7b-chat.ggmlv3.q4_0.bin


## From the following link:
https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGML/tree/main
```

```bash
# run the following command
python store_index.py
```

```bash
# Finally run the following command
python app.py
```

Now,
```bash
open up localhost:
```


### Techstack Used:

- Python
- LangChain
- Flask
- Meta Llama2
- Pinecone
