import os
import requests
from langchain_core.embeddings import Embeddings

class OpenRouterEmbeddings(Embeddings):
    def __init__(self, model_name: str = "openai/text-embedding-3-small"):
        self.model_name = model_name
        self.api_key = os.environ["OPENROUTER_API_KEY"]
        self.url = "https://openrouter.ai/api/v1/embeddings"

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of documents by calling OpenRouter directly."""
        # Ensure all texts are clean strings
        string_texts = [str(text) if not hasattr(text, 'page_content') else text.page_content for text in texts]
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.model_name,
            "input": string_texts
        }
        
        response = requests.post(self.url, headers=headers, json=payload)
        
        # Raise an exception if OpenRouter returns an error code (400, 401, 500, etc.)
        response.raise_for_status()
        
        data = response.json()
        
        # Extract the float array safely
        return [item["embedding"] for item in data["data"]]

    def embed_query(self, text: str) -> list[float]:
        """Embed a single search query."""
        embeddings = self.embed_documents([text])
        return embeddings[0]
