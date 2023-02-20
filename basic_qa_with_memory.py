import logging
import sys
import os

from langchain.agents import Tool
from langchain.chains.conversation.memory import ConversationBufferMemory, ConversationSummaryBufferMemory
from langchain import OpenAI
from langchain.agents import initialize_agent
from llama_index import GPTSimpleVectorIndex

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
        name = "GPT Index",
        func=lambda q: str(index.query(q)),
        description="useful for when you want to answer questions about the author. The input to this tool should be a complete english sentence.",
        return_direct=True
    ),
]

# memory = ConversationBufferMemory(memory_key="chat_history")
memory = ConversationSummaryBufferMemory(llm=OpenAI(), memory_key="chat_history")
print("Initialized Memory...")

llm=OpenAI(temperature=0)
print("Initialized LLM...")

agent_chain = initialize_agent(tools, llm, agent="conversational-react-description", memory=memory)
print("Initialized Agent...")

while True:
	print("Your input:")
	input_text = input()
	print(agent_chain.run(input=input_text))