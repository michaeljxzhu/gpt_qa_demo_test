import logging
import sys
import os

from langchain.agents import Tool
from langchain.chains.conversation.memory import ConversationBufferMemory, ConversationSummaryBufferMemory
from langchain import OpenAI
from llama_index import GPTSimpleVectorIndex
from langchain.agents.conversational.base import ConversationalAgent
from langchain.agents.agent import AgentExecutor

from health_conversational_agent import HealthConversationalAgent

# note: set Logging to DEBUG for more detailed outputs
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

# KeyError if OPENAPI_API_KEY not set
os.environ["OPENAI_API_KEY"]

# create gpt-index index
print(f"Attempting to load GPTSimpleVectorIndex")
index = GPTSimpleVectorIndex.load_from_disk('wikipedia_index.json')
print(f"Loaded GPTSimpleVectorIndex")

# define langchain tool calling into gpt-index index
tools = [
    Tool(
        name = "Medical Information",
        func=lambda q: str(index.query(q)),
        description="This should be used whenever the human asks a question about cancer, medicine, and health",
        return_direct=True # explore setting this to false
    ),
]

# memory = ConversationBufferMemory(memory_key="chat_history")
memory = ConversationSummaryBufferMemory(llm=OpenAI(), memory_key="chat_history")
print("Initialized Memory...")

llm=OpenAI(temperature=0)
print("Initialized LLM...")

agent_obj = HealthConversationalAgent.from_llm_and_tools(
    llm,
    tools,
    memory=memory,
    verbose=True
)
agent = AgentExecutor.from_agent_and_tools(
    agent=agent_obj,
    tools=tools,
    memory=memory,
    verbose=True
)	
print("Initialized Agent...")

while True:
	print("Your input:")
	input_text = input()
	agent(input_text)