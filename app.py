import streamlit as st
import json
import re
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-001", temperature=0.3)


persist_directory = "chroma_db"
db = Chroma(persist_directory=persist_directory, embedding_function=embedding_model)


import streamlit as st
import json
import re

def data_extraction(content):
    assessment_name_pattern = r"name:\s*(.*)"
    description_pattern = r"Description:\s*(.*)"
    test_type_pattern = r"Test Type:\s*(.*)"
    job_levels_pattern = r"Job Levels:\s*(.*)"
    assessment_length_pattern = r"Assessment Length:\s*(.*)"
    remote_testing_pattern = r"Remote Testing:\s*(.*)"
    adaptive_irt_pattern = r"Adaptive/IRT:\s*(.*)"
    url_pattern = r"url:\s*(.*)"

    assessment_name = re.search(assessment_name_pattern, content)
    description = re.search(description_pattern, content)
    test_type = re.search(test_type_pattern, content)
    job_levels = re.search(job_levels_pattern, content)
    assessment_length = re.search(assessment_length_pattern, content)
    remote_testing = re.search(remote_testing_pattern, content)
    adaptive_irt = re.search(adaptive_irt_pattern, content)
    url = re.search(url_pattern, content)

    assessment_name = assessment_name.group(1) if assessment_name else "N/A"
    description = description.group(1) if description else "N/A"
    test_type = test_type.group(1) if test_type else "N/A"
    job_levels = job_levels.group(1) if job_levels else "N/A"
    assessment_length = assessment_length.group(1) if assessment_length else "N/A"
    remote_testing = remote_testing.group(1) if remote_testing else "N/A"
    adaptive_irt = adaptive_irt.group(1) if adaptive_irt else "N/A"
    url = url.group(1) if url else "N/A"

    return {
        "Assessment Name": assessment_name,
        "Description": description,
        "Test Type": test_type,
        "Job Levels": job_levels,
        "Assessment Length": assessment_length,
        "Remote Testing": remote_testing,
        "Adaptive/IRT": adaptive_irt,
        "url": url
    }

st.title("SHL Assessment Recommender")

query = st.text_input("Enter your query (e.g. sales management):")
prompt = f"Extract skills, job roles, time limit, or any specific test type from this query: '{query}' and do not add anything more from you"
    
if query:
    query_features = llm.invoke(prompt).content
    st.write(query_features)

    query_embedding = embedding_model.embed_query(query_features)
    results = db.similarity_search_by_vector(query_embedding, k=10)

    recommendations = []

    st.subheader("Top Recommendations")
    for i, doc in enumerate(results, 1):
        content = doc.page_content
        extracted_data = data_extraction(content)
        
        st.markdown(f"###  Result {i}")
        st.write(f"**Assessment Name:** {extracted_data['Assessment Name']}")
        st.write(f"**Description:** {extracted_data['Description']}")
        st.write(f"**Test Type:** {extracted_data['Test Type']}")
        st.write(f"**Job Levels:** {extracted_data['Job Levels']}")
        st.write(f"**Length:** {extracted_data['Assessment Length']}")
        st.write(f"**Remote Testing:** {extracted_data['Remote Testing']}")
        st.write(f"**Adaptive/IRT:** {extracted_data['Adaptive/IRT']}")
        st.write(f"**URL:** {extracted_data['url']}")

        explanation = llm.invoke(
            f"Explain why the assessment '{extracted_data['Assessment Name']}' is suitable for a '{extracted_data['Job Levels']}' role. Keep it short."
        )
        st.write(f"**Explanation:** {explanation.content}")

        recommendations.append(extracted_data)

    json_str = json.dumps(recommendations, indent=2)
    st.download_button(
        label="Download Results as JSON",
        data=json_str,
        file_name="assessment_recommendations.json",
        mime="application/json"
    )
