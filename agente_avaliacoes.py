import requests
import json

http_reviews = requests.get("https://apis.codante.io/api/reviews-api/reviews")
dict_reviews = json.loads(http_reviews.content.decode("utf-8"))
list_review = []

for review in dict_reviews["data"]:
  list_review.append(review["description"])

#Desenvolva, a partir daqui, um chain para um agente que avalia se as revisões recebidas são positivas ou negativas.
#Utilize todos os componentes do LangChain que trabalhamos até este momento para isso.

from langchain_groq import ChatGroq
from dotenv import load_dotenv
from os import getenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

GROQ_API_KEY=getenv("GROQ_API_KEY")
print(GROQ_API_KEY)
PROMPT_TEMPLATE="""
Avalie a seguinte revisão se a mesma é positiva ou negativa. Utilize busca por padrões que digam se o review é positivo ou negativo.
Responda apenas um token 0 ou 1: 0: review negativo, 1: review positivo.

Exemplo 1:
Review: "Este produto tem boa qualidade de construção, mas o som estoura nos agudos e médios e não fica bom"
Resposta: 0

Exemplo 2:
Review: "Gostei, o som é excelente!"
Resposta: 1

=====

Review: {review}.
Resposta:
"""

llm = ChatGroq(
    groq_api_key = GROQ_API_KEY,
    model        = "qwen/qwen3.8-27b",
    temperature  = 0.0,
    max_tokens = 1,
)

prompt = PromptTemplate(
    input_variables=["review"],
    template=PROMPT_TEMPLATE
)
output_parser = StrOutputParser()

chain = prompt | llm | output_parser

for review in list_review:

  resposta = chain.invoke({"review":review}).replace("\n", "").strip()
  print("Review: " + review + "\nLLM: " + ("negativo" if resposta == "0" else "positivo") + "\n")
