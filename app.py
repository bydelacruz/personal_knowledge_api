"""
app GUI
"""

import requests
import streamlit as st

# Point to the Backend API
API_URL = "http://localhost:8000"

st.set_page_config(page_title="Knowledge Base", layout="wide")

st.title("🧠 Knowledge Base")
st.markdown("Ask questions about your notes, PDFs, and data.")

# --- HELPER FUNCTIONS ---
# We define these functions to handle the logic BEFORE the page reloads.


def save_note():
    # 1. Get data from Session State
    payload = {
        "topic": st.session_state.note_topic,
        "rating": st.session_state.note_rating,
        "tags": [t.strip() for t in st.session_state.note_tags.split(",")],
    }

    try:
        res = requests.post(f"{API_URL}/notes", json=payload)
        if res.status_code == 201:
            st.toast("✅ Note saved successfully!")
            # 2. CLEAR STATE (This happens before the widgets are drawn again)
            st.session_state.note_topic = ""
            st.session_state.note_rating = 5
            st.session_state.note_tags = "general"
        else:
            st.error(f"Error: {res.text}")
    except Exception as e:
        st.error(f"Connection Failed: {e}")


def process_pdf():
    uploaded_file = st.session_state.pdf_uploader
    if uploaded_file is None:
        return

    files = {"file": (uploaded_file.name, uploaded_file, "application/pdf")}
    try:
        res = requests.post(f"{API_URL}/upload-pdf", files=files)
        if res.status_code == 200:
            data = res.json()
            st.toast(f"✅ PDF Processed! {data['chunks_processed']} chunks added.")
            # CLEAR STATE
            st.session_state.pdf_uploader = None
        else:
            st.error(f"Error: {res.text}")
    except Exception as e:
        st.error(f"Upload Failed: {e}")


# --- SIDEBAR: ADD DATA ---
with st.sidebar:
    st.header("Add Knowledge")

    # Tab 1: Manual Note
    with st.expander("📝 Add Text Note"):
        # We bind the widgets to session state keys
        st.text_input("Topic/Content", key="note_topic")
        st.slider("Importance Rating", 1, 10, 5, key="note_rating")
        st.text_input("Tags (comma separated)", "general", key="note_tags")

        # WE CALL THE FUNCTION IN 'on_click'
        st.button("Save Note", on_click=save_note)

    # Tab 2: PDF Upload
    with st.expander("📂 Upload PDF"):
        st.file_uploader("Choose a PDF", type="pdf", key="pdf_uploader")

        # WE CALL THE FUNCTION IN 'on_click'
        st.button("Process PDF", on_click=process_pdf)

# --- MAIN AREA: CHAT INTERFACE ---

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("What would you like to know?"):
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

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

                full_response = f"{answer}\n\n"

                if sources:
                    with st.expander("📚 View Sources"):
                        for i, source in enumerate(sources):
                            st.markdown(f"**Source {i + 1}:** {source}")

                message_placeholder.markdown(full_response)
                st.session_state.messages.append(
                    {"role": "assistant", "content": full_response}
                )
            else:
                message_placeholder.error(f"API Error: {res.text}")

        except Exception as e:
            message_placeholder.error(f"Connection Error: Is the backend running? {e}")
