import streamlit as st
import random
import time
import base64
import os
from dotenv import load_dotenv
from langchain.schema import HumanMessage
from langchain_community.vectorstores  import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
import google.generativeai as genai

LOGO_IMAGE = "deloitte-logo-w.png"

st.markdown(
    """
    <style>
    .container {
        display: flex;
    }
    .logo-text {
        font-weight:700 !important;
        font-size:30px !important;
        color: #ffffff !important;
        padding-top: 37px;
    }
    .logo-img {
        float:right;
        width: 125px;
        height: 125px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown(
    f"""
    <div class="container">
        <p class="logo-text">DocuSeek by </p>
        <img class="logo-img" src="data:image/png;base64,{base64.b64encode(open(LOGO_IMAGE, "rb").read()).decode()}">
        
    </div>
    """,
    unsafe_allow_html=True
)


# Load environment variables

GOOGLE_API_KEY = 'AIzaSyD6potSTkJcaEHJD4mUaAmNFA2BVDwY7oI'

os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

genai.configure(api_key = GOOGLE_API_KEY)

CHROMA_PATH = "chroma"

PROMPT_TEMPLATE = """

Given the following context passages retrieved by a similarity search, answer the user's question using ONLY the information from the context. For your response, always provide the source (e.g., filename, document title, or passage ID) from which the answer is derived.

Response format:
Source: <source of the answer>

Answer: <your answer based on the context>


Context:
{context}
 
Question:
{question}

"""

def main():
      
    def response_generator(question):

        query_text = question

    # Setup embedding
        embedding_function = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            task_type="RETRIEVAL_QUERY"
        )

        # Load DB
        db = Chroma(
            persist_directory=CHROMA_PATH,
            embedding_function=embedding_function
        )
    
        # Search
        results = db.similarity_search_with_relevance_scores(query_text, k=7)
        print("=== Result ===")
        print(results)
        if len(results) == 0 or results[0][1] < 0.2:
            print("Unable to find matching results.")
            return

        # Format context
        context_text = "\n\n---\n\n".join([doc.page_content for doc, _ in results])
        prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
        prompt = prompt_template.format(context=results, question=query_text)
        print("=== Prompt ===")
        print(prompt)

        # Use Gemini chat model
        model = ChatGoogleGenerativeAI(model="models/gemini-2.0-flash")
        response = model.invoke([HumanMessage(content=prompt)])

        # Output
        sources = [doc.metadata.get("source", None) for doc, _ in results]
        sourcesUnique = list(dict.fromkeys(sources))
        print(f"\nResponse: {response.content}")
        print(f"Sources: {sourcesUnique}")

        response = response.content

        separator = "; "
        uniqueSources = separator.join(sourcesUnique)

        #response = '[Sources: ' + uniqueSources + ']\n\n' + response
    
        yield response 

# Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

# Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# Accept user input
    if question := st.chat_input("Ask your question here!"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": question})
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(question)

        # Display assistant response in chat message container

        with st.chat_message("assistant"):
            response = st.write_stream(response_generator(question))


    # Add assistant response to chat history
        st.session_state.messages.append(
        {
            "role": "assistant", 
            "content": response
        }
        )
if __name__ == "__main__":
    main()

