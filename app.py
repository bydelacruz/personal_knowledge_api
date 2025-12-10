"""
app GUI
"""

import requests
import streamlit as st

# Point to the Backend API
# If running locally with Docker/Render: use localhost
# If running purely local (no Docker): use http://127.0.0.1:8000
API_URL = "http://localhost:8000"

st.set_page_config(page_title="Knowledge Base", layout="wide")

st.title("🧠 Knowledge Base")
st.markdown("Ask questions about your notes, PDFs, and data.")

# --- SIDEBAR: ADD DATA ---
with st.sidebar:
    st.header("Add Knowledge")

    # Tab 1: Manual Note
    # We assign a key to the expander state could be tricky,
    # but st.rerun() usually resets expanders to default (closed).
    with st.expander("📝 Add Text Note"):
        # 1. ADD KEYS to every input so we can wipe them later
        topic = st.text_input("Topic/Content", key="note_topic")
        rating = st.slider("Importance Rating", 1, 10, 5, key="note_rating")
        tags = st.text_input("Tags (comma separated)", "general", key="note_tags")

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

                    # 2. RESET SESSION STATE
                    st.session_state.note_topic = ""  # Clear text
                    st.session_state.note_rating = 5  # Reset slider
                    st.session_state.note_tags = "general"  # Reset tags

                    # 3. RERUN APP (Refreshes UI and closes expander)
                    st.rerun()
                else:
                    st.error(f"Error: {res.text}")
            except Exception as e:
                st.error(f"Connection Failed: {e}")

    # Tab 2: PDF Upload
    with st.expander("📂 Upload PDF"):
        # ADD KEY to file uploader
        uploaded_file = st.file_uploader("Choose a PDF", type="pdf", key="pdf_uploader")

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

                            # RESET SESSION STATE
                            # Setting file_uploader key to None clears it
                            st.session_state.pdf_uploader = None

                            # RERUN APP
                            st.rerun()
                        else:
                            st.error(f"Error: {res.text}")
                    except Exception as e:
                        st.error(f"Upload Failed: {e}")

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
