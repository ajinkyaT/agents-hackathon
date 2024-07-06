# LangGraph FastAPI POC
An agentic app built using langraph and FastAPI in python

## First time setup

1. Install poetry and poetry-dotenv-plugin, run below two commands in the terminal 
``` shell
$ pip install poetry poetry-dotenv-plugin
$ poetry self add poetry-dotenv-plugin
```

2. Go to root of the project and run below coomands. This will install all the needed python packages 
``` shell
$ poetry update
```

3. Paste your secret keys needed to run LLM models in .env file in the root of the project

``` text
# Example content of .env file
GROQ_API_KEY="gsk_gn**"
ANTHROPIC_API_KEY="sk-**"
```

Note you need to change code on line 129, in package unstructured, partition, csv.py in method get_delimiter
``` python
data = "".join(f.readline())
```

4. Run python app using poetry
``` shell
$ poetry run python app.py
```
