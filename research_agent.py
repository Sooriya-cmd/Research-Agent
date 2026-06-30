from dotenv import load_dotenv
from pydantic import BaseModel
from langchain.agents import create_agent
from langchain_core.utils.uuid import uuid7
from langchain_google_genai import ChatGoogleGenerativeAI
from tools import search_tool, wiki_tool, save_tool

load_dotenv()


class ResearchResponse(BaseModel):
    topic: str
    summary: str
    sources: list[str]
    tools_used: list[str]


SYSTEM_PROMPT = """
You are an ai assistant designed to help users research topics efficiently and clearly ,use Boss whenever you need to refer to the user. Your role is to provide clear, accurate, and helpful answers in a calm, precise tone. You have access to the following tools: DuckDuckGoSearchRun, WikipediaQueryRun, and save_text_to_file. Use these tools as needed to gather information and provide a comprehensive response.
Answer the user query and use necessary tools.
"""

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", max_retries=4)

tools = [
    search_tool,
    save_tool
]

agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt=SYSTEM_PROMPT,
    response_format=ResearchResponse,
)

query = input("What can I help you research? ")
config = {"configurable": {"thread_id": str(uuid7())}}

raw_response = agent.invoke(
    {"messages": [{"role": "user", "content": query}]},
    config=config,
)

try:
    structured_response = raw_response["structured_response"]
    print(structured_response)
except Exception as e:
    print("Error parsing response", e, "Raw Response - ", raw_response)