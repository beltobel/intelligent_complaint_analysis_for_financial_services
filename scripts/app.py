import streamlit as st
from sentence_transformers import SentenceTransformer
import chromadb
from transformers import pipeline

# --- Load models and vector store ---
@st.cache_resource
def load_models():
    embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    client = chromadb.PersistentClient(path="./vector_store")
    collection = client.get_or_create_collection(name="cfpb_complaints")
    llm = pipeline("text-generation", model="google/flan-t5-small", max_new_tokens=256)
    return embedding_model, collection, llm

embedding_model, collection, llm = load_models()

PROMPT_TEMPLATE = (
    "You are a financial analyst assistant for CrediTrust. "
    "Your task is to answer questions about customer complaints. "
    "Use the following retrieved complaint excerpts to formulate your answer. "
    "If the context doesn't contain the answer, state that you don't have enough information.\n\n"
    "Context:\n{context}\n\nQuestion: {question}\nAnswer:"
)

def retrieve_relevant_chunks(question, k=5):
    question_embedding = embedding_model.encode(question).tolist()
    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=k,
        include=['documents', 'metadatas']
    )
    docs = results['documents'][0]
    metas = results['metadatas'][0]
    return docs, metas

def generate_answer(question, context_chunks):
    context = "\n".join(context_chunks)
    prompt = PROMPT_TEMPLATE.format(context=context, question=question)
    response = llm(prompt)
    st.write("DEBUG: LLM raw response:", response)  # Show raw output in Streamlit
    answer = response[0].get('generated_text', '').split("Answer:")[-1].strip()
    return answer

# --- Streamlit UI ---
st.set_page_config(page_title="CrediTrust Complaint Analyst", page_icon="💬")
st.title("💬 CrediTrust Complaint Analyst")
st.write("Ask a question about customer complaints. The AI will answer using real complaint excerpts.")

if "history" not in st.session_state:
    st.session_state.history = []

with st.form(key="qa_form"):
    user_question = st.text_input("Your question:", key="question_input")
    submit = st.form_submit_button("Ask")
    clear = st.form_submit_button("Clear")

if clear:
    st.session_state.history = []
    st.experimental_rerun()

if submit and user_question.strip():
    with st.spinner("Retrieving and generating answer..."):
        retrieved_chunks, _ = retrieve_relevant_chunks(user_question, k=5)
        st.write("DEBUG: Retrieved Chunks:", retrieved_chunks)  # Add this line
        answer = generate_answer(user_question, retrieved_chunks)
        st.session_state.history.append({
            "question": user_question,
            "answer": answer,
            "sources": retrieved_chunks[:2]
        })
# Display conversation history
for entry in reversed(st.session_state.history):
    st.markdown(f"**You:** {entry['question']}")
    st.markdown(f"**AI:** {entry['answer']}")
    with st.expander("Show sources"):
        for i, src in enumerate(entry['sources']):
            st.markdown(f"**Source {i+1}:** {src}")