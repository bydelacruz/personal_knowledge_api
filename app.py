"""
app GUI
"""

import requests
import streamlit as st

# Point to FastAPI backend
API_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="Personal Knowledge Brain", layout="wide")
st.title("🧠 Personal Knowledge Brain")
st.markdown("Ask questions about your notes, PDFs, and data.")

# --- SIDEBAR: ADD DATA ---
with st.sidebar:
    st.header("Add Knowledge")

    # Tab 1: Manual Note
    with st.expander("📝 Add Text Note"):
        topic = st.text_input("Topic/Content")
        rating = st.slider("Importance Rating", 1, 10, 5)
        tags = st.text_input("Tags (comma separated)", "general")

        if st.button("Save Note"):
            payload = {
                "topic": topic,
                "rating": rating,
                "tags": [t.strip() for t in tags.split(",")],
            }
            try:
                res = requests.post(f"{API_URL}/notes", json=payload)
                if res.status_code == 201:
                    st.success("Note saved!")
                else:
                    st.error(f"Error: {res.text}")
            except Exception as e:
                st.error(f"Connection Failed: {e}")

    # Tab 2: PDF Upload
    with st.expander("📂 Upload PDF"):
        uploaded_file = st.file_uploader("Choose a PDF", type="pdf")
        if uploaded_file is not None:
            if st.button("Process PDF"):
                with st.spinner("Ingesting and Chunking..."):
                    files = {
                        "file": (uploaded_file.name, uploaded_file, "application/pdf")
                    }
                    try:
                        res = requests.post(f"{API_URL}/upload-pdf", files=files)
                        if res.status_code == 200:
                            data = res.json()
                            st.success(
                                f"""Success! Split into {
                                    data["chunks_processed"]
                                } chunks."""
                            )
                        else:
                            st.error(f"Error: {res.text}")
                    except Exception as e:
                        st.error(f"Upload Failed: {e}")
# --- MAIN AREA: CHAT INTERFACE ---

# Initialize chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("What would you like to know?"):
    # 1. Display user message
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # 2. Call the backend API
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        message_placeholder.markdown("Thinking...")

        try:
            payload = {"question": prompt}
            res = requests.post(f"{API_URL}/ask", json=payload)

            if res.status_code == 200:
                data = res.json()
                answer = data["answer"]
                sources = data["sources"]

                # Format the answer
                full_response = f"{answer}\n\n"

                # Add sources dropdown
                if sources:
                    with st.expander("📚 View Sources"):
                        for i, source in enumerate(sources):
                            st.markdown(f"**Source {i + 1}:** {source}")

                message_placeholder.markdown(full_response)

                # Add to history
                st.session_state.messages.append(
                    {"role": "assistant", "content": full_response}
                )
            else:
                message_placeholder.error(f"API Error: {res.text}")
        except Exception as e:
            message_placeholder.error(f"Connection Error: Is the backend running? {e}")
