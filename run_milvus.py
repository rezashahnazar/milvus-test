from openai import OpenAI
from pymilvus import MilvusClient
import os
from dotenv import dotenv_values
from typing import List, Dict

config = dotenv_values(".env")

OPENAI_API_KEY = config["OPENAI_API_KEY"]
OPENAI_BASE_URL = config["OPENAI_BASE_URL"]


class SemanticSearch:
    def __init__(self):
        # Initialize OpenAI client
        self.openai_client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)

        # Initialize Milvus client (standalone)
        self.milvus_client = MilvusClient(
            uri="http://localhost:19530",  # Added 'http://' prefix to URI
            # user=os.getenv("MILVUS_USER", ""),  # Optional
            # password=os.getenv("MILVUS_PASSWORD", ""),  # Optional
        )

        # Constants
        self.COLLECTION_NAME = "semantic_search_demo"
        self.EMBEDDING_MODEL = "text-embedding-3-small"
        self.DIMENSION = 1536  # Dimension for text-embedding-3-small model

        # Create collection if it doesn't exist
        self._initialize_collection()

    def _initialize_collection(self):
        """Initialize Milvus collection if it doesn't exist"""
        if self.milvus_client.has_collection(self.COLLECTION_NAME):
            print(f"Collection {self.COLLECTION_NAME} already exists")
            return

        # Create collection with 'vector' field name instead of 'embedding'
        self.milvus_client.create_collection(
            collection_name=self.COLLECTION_NAME,
            dimension=self.DIMENSION,
            primary_field="id",
            vector_field="vector",  # Changed from 'embedding' to 'vector'
        )
        print(f"Created collection {self.COLLECTION_NAME}")

    def _get_embedding(self, text: str) -> List[float]:
        """Get embedding for a text using OpenAI's API"""
        response = self.openai_client.embeddings.create(
            input=text, model=self.EMBEDDING_MODEL
        )
        return response.data[0].embedding

    def add_documents(self, documents: List[Dict[str, str]]):
        """
        Add documents to the semantic search index
        documents: List of dicts with 'id' and 'text' keys
        """
        for doc in documents:
            # Get embedding for the document
            embedding = self._get_embedding(doc["text"])

            # Insert into Milvus without wrapping embedding in a list
            self.milvus_client.insert(
                collection_name=self.COLLECTION_NAME,
                data={
                    "id": doc["id"],
                    "vector": embedding,  # Removed the extra list wrapper
                    "text": doc["text"],
                },
            )

        print(f"Added {len(documents)} documents to the index")

    def search(self, query: str, limit: int = 3) -> List[Dict]:
        """
        Search for similar documents
        query: Search query
        limit: Number of results to return
        """
        # Get embedding for the query
        query_embedding = self._get_embedding(query)

        # Search in Milvus
        results = self.milvus_client.search(
            collection_name=self.COLLECTION_NAME,
            data=[query_embedding],
            limit=limit,
            output_fields=["text"],
        )

        return results[0]  # Return first (and only) query's results


def main():
    # Initialize semantic search
    search_engine = SemanticSearch()

    # Example documents
    documents = [
        {
            "id": 1,
            "text": "The Internet of Things (IoT) represents a network of interconnected devices that collect and exchange data through embedded sensors and software, enabling smart homes, industrial automation, and urban infrastructure management.",
        },
        {
            "id": 2,
            "text": "Quantum computing leverages quantum mechanical phenomena like superposition and entanglement to perform computations that would be practically impossible for traditional computers, potentially revolutionizing cryptography and drug discovery.",
        },
        {
            "id": 3,
            "text": "Edge computing brings computation and data storage closer to the location where it is needed, reducing latency and bandwidth usage while improving response times and privacy in applications like autonomous vehicles and smart cities.",
        },
        {
            "id": 4,
            "text": "Blockchain technology provides a decentralized, immutable ledger that enables secure transactions and record-keeping without intermediaries, finding applications beyond cryptocurrency in supply chain management, voting systems, and digital identity verification.",
        },
        {
            "id": 5,
            "text": "Computer vision systems utilize deep learning algorithms to process and analyze visual information from the world, enabling applications like facial recognition, autonomous navigation, medical image analysis, and quality control in manufacturing.",
        },
        {
            "id": 6,
            "text": "Augmented Reality (AR) overlays digital information onto the physical world, enhancing user experiences in fields like education, healthcare, manufacturing, and entertainment through sophisticated spatial computing and computer vision techniques.",
        },
        {
            "id": 7,
            "text": "Natural Language Processing has evolved to understand context, sentiment, and nuance in human communication, powering applications like machine translation, chatbots, voice assistants, and automated content analysis.",
        },
        {
            "id": 8,
            "text": "5G networks provide ultra-low latency and high bandwidth connectivity, enabling real-time applications like remote surgery, holographic communications, and massive IoT deployments in smart cities and industrial settings.",
        },
        {
            "id": 9,
            "text": "Robotic Process Automation (RPA) combines artificial intelligence and machine learning to automate repetitive business processes, improving efficiency and accuracy in tasks like data entry, customer service, and financial reconciliation.",
        },
        {
            "id": 10,
            "text": "Cybersecurity in the modern era employs artificial intelligence to detect and prevent sophisticated threats, using behavioral analysis, anomaly detection, and predictive modeling to protect digital assets and infrastructure.",
        },
    ]

    # Add documents to the index
    search_engine.add_documents(documents)

    while True:
        # Get query from user
        query = input("\nEnter your search query (or 'quit' to exit): ").strip()

        if query.lower() in ["quit", "exit", "q"]:
            break

        if not query:
            continue

        # Perform search
        results = search_engine.search(query, limit=2)

        print("\nSearch Results for:", query)
        print("-" * 50)
        for result in results:
            print(
                f"Score: {-result['distance']:.4f}"
            )  # Convert distance to similarity score
            print(f"Text: {result['entity']['text']}")
            print("-" * 50)


if __name__ == "__main__":
    main()
