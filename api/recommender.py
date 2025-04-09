import re
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

# Initialize the embedding model, LLM, and Chroma database
embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-001", temperature=0.3)
db = Chroma(persist_directory="chroma_db", embedding_function=embedding_model)

def extract_fields(content: str):
    fields = {
        "name": r"name:\s*(.*)",
        "description": r"Description:\s*(.*)",
        "test_type": r"Test Type:\s*(.*)",
        "job_levels": r"Job Levels:\s*(.*)",
        "duration": r"Assessment Length:\s*(.*)",
        "remote_support": r"Remote Testing:\s*(.*)",
        "adaptive_support": r"Adaptive/IRT:\s*(.*)",
        "url": r"url:\s*(.*)",
    }

    extracted = {}
    for key, pattern in fields.items():
        match = re.search(pattern, content)
        extracted[key] = match.group(1).strip() if match else "N/A"

    return extracted

def get_recommendations(query: str):
    """Get recommendations based on the query."""
    prompt = f"Extract skills, job roles, time limit, or any specific test type from this query: '{query}' and do not add anything more from you"
    query_features = llm.invoke(prompt).content

    # Embed the query and perform a similarity search
    query_embedding = embedding_model.embed_query(query_features)
    results = db.similarity_search_by_vector(query_embedding, k=10)

    formatted_results = []
    for doc in results:
        data = extract_fields(doc.page_content)
        formatted_results.append({
            "url": data["url"],
            "adaptive_support": "Yes" if "yes" in data["adaptive_support"].lower() else "No",
            "description": data["description"],
            "duration": int(re.findall(r'\d+', data["duration"])[0]) if re.findall(r'\d+', data["duration"]) else 0,
            "remote_support": "Yes" if "yes" in data["remote_support"].lower() else "No",
            "test_type": [x.strip() for x in data["test_type"].split(",") if x.strip()]
        })

    return formatted_results