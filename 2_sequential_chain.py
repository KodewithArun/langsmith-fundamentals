from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os


os.environ["LANGCHAIN_PROJECT"] = "Sequential Chain"
load_dotenv()

prompt1 = PromptTemplate(
    template='Generate a detailed report on {topic}',
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template='Generate a 5 pointer summary from the following text \n {text}',
    input_variables=['text']
)

model = ChatGroq(model="openai/gpt-oss-120b")

parser = StrOutputParser()

chain = prompt1 | model | parser | prompt2 | model | parser

config={
    "run_name": "Sequential Chain",
    "tags": ["Sequential Chain", "Simple Chain","Nov 13","Langsmith Launch"],
    "metadata": {
        "topic": "Unemployment in Nepal"
    }
}

result = chain.invoke({'topic': 'Unemployment in Nepal'}, config=config)

print(result)
