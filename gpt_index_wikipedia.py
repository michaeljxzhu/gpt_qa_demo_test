from llama_index import GPTSimpleVectorIndex, SimpleDirectoryReader, WikipediaReader
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.embeddings.openai import OpenAIEmbeddings
from llama_index import LangchainEmbedding
import os
import sys

from wikipedia_pages import get_links_for_wikipedia_page, WIKIPEDIA_PAGES_LIST

# KeyError if OPENAPI_API_KEY not set
os.environ["OPENAI_API_KEY"]
# Resolves warning: huggingface/tokenizers: The current process just got forked,
# after parallelism has already been used. Disabling parallelism to avoid deadlocks
os.environ["TOKENIZERS_PARALLELISM"] = "false"

VECTOR_INDEX_PATH = 'wikipedia_index.json'

print(f"Attempting to load embeddings")
# other options: HuggingFaceEmbeddings, CohereEmbeddings, etc...
embed_model = LangchainEmbedding(OpenAIEmbeddings())
print(f"Loading embeddings")

print(f"Attempting to load index")
try:
	index = GPTSimpleVectorIndex.load_from_disk(
		VECTOR_INDEX_PATH,
		embed_model=embed_model
	)
	print(f"Index loaded from disk")
except OSError as e:
	print(f"Unable to open {VECTOR_INDEX_PATH}: {e}", file=sys.stderr)
	print(f"Creating new index from scratch, using list: {WIKIPEDIA_PAGES_LIST}")
	index = GPTSimpleVectorIndex(
		[],
		embed_model=embed_model
	)	
	print(f"Created in-memory index")
	for page_name in WIKIPEDIA_PAGES_LIST:
		try:
			documents = WikipediaReader().load_data(
				pages=[page_name],
			)
			for document in documents:
				index.insert(
					document
				)
			print(f"Inserted document for {page_name}")
		except Exception:
			continue

	index.save_to_disk(VECTOR_INDEX_PATH)
	print(f"Saved in-memory index to file {VECTOR_INDEX_PATH}")

while True:
	print(f"What would you like to ask?")
	user_input = input()
	response = index.query(
		user_input
	)
	print(response)
