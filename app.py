"""
app GUI
"""

import requests
import streamlit as st

# Point to the Backend API
API_URL = "http://localhost:8000"

st.set_page_config(page_title="Personal Knowledge Brain", layout="wide")

# --- AUTHENTICATION STATE ---
if "token" not in st.session_state:
    st.session_state.token = None
if "user" not in st.session_state:
    st.session_state.user = None

# --- AUTH FUNCTIONS ---


def login_user(username, password):
    try:
        # OAuth2 expects form data, not JSON
        payload = {"username": username, "password": password}
        res = requests.post(f"{API_URL}/token", data=payload)

        if res.status_code == 200:
            data = res.json()
            st.session_state.token = data["access_token"]
            st.session_state.user = username
            st.toast(f"Welcome back, {username}!")
            st.rerun()
        else:
            st.error("Invalid username or password")
    except Exception as e:
        st.error(f"Connection Error: {e}")


def register_user(username, password):
    try:
        payload = {"username": username, "password": password}
        res = requests.post(f"{API_URL}/register", json=payload)

        if res.status_code == 201:
            st.success("Account created! You can now log in.")
        else:
            st.error(f"Error: {res.text}")
    except Exception as e:
        st.error(f"Connection Error: {e}")


def logout():
    st.session_state.token = None
    st.session_state.user = None


# --- THE GATEKEEPER ---
if not st.session_state.token:
    # === LOGIN / REGISTER SCREEN ===
    st.title("🔒 Login to Brain")

    tab1, tab2 = st.tabs(["Login", "Register"])

    with tab1:
        l_user = st.text_input("Username", key="l_user")
        l_pass = st.text_input("Password", type="password", key="l_pass")
        if st.button("Login"):
            login_user(l_user, l_pass)

    with tab2:
        r_user = st.text_input("New Username", key="r_user")
        r_pass = st.text_input("New Password", type="password", key="r_pass")
        if st.button("Create Account"):
            register_user(r_user, r_pass)

    st.stop()  # STOP EXECUTION HERE if not logged in

# === THE MAIN APP (Only runs if token exists) ===
# Header with Logout
col1, col2 = st.columns([8, 1])
with col1:
    st.title(f"🧠 {st.session_state.user}'s Brain")
with col2:
    st.button("Logout", on_click=logout)

# --- HELPER FUNCTIONS (Authenticated) ---
# We add headers={"Authorization": f"Bearer {token}"}


def get_auth_headers():
    return {"Authorization": f"Bearer {st.session_state.token}"}


def save_note():
    payload = {
        "topic": st.session_state.note_topic,
        "rating": st.session_state.note_rating,
        "tags": [t.strip() for t in st.session_state.note_tags.split(",")],
    }

    try:
        # ADD HEADERS HERE
        res = requests.post(
            f"{API_URL}/notes", json=payload, headers=get_auth_headers()
        )
        if res.status_code == 201:
            st.toast("✅ Note saved successfully!")
            st.session_state.note_topic = ""
            st.session_state.note_rating = 5
            st.session_state.note_tags = "general"
        else:
            st.error(f"Error: {res.text}")
    except Exception as e:
        st.error(f"Connection Failed: {e}")


def process_pdf():
    current_key = f"pdf_uploader_{st.session_state.uploader_key}"
    if current_key not in st.session_state:
        return
    uploaded_file = st.session_state[current_key]
    if uploaded_file is None:
        return

    files = {"file": (uploaded_file.name, uploaded_file, "application/pdf")}
    try:
        # ADD HEADERS HERE
        res = requests.post(
            f"{API_URL}/upload-pdf", files=files, headers=get_auth_headers()
        )
        if res.status_code == 200:
            data = res.json()
            st.toast(f"✅ PDF Processed! {data['chunks_processed']} chunks added.")
            st.session_state.uploader_key += 1
        else:
            st.error(f"Error: {res.text}")
    except Exception as e:
        st.error(f"Upload Failed: {e}")


# --- INITIALIZE STATE ---
if "init" not in st.session_state:
    st.session_state.init = True
    st.session_state.note_topic = ""
    st.session_state.note_rating = 5
    st.session_state.note_tags = "general"
    st.session_state.uploader_key = 0
    st.session_state.messages = []

# --- SIDEBAR: ADD DATA ---
with st.sidebar:
    st.header("Add Knowledge")
    with st.expander("📝 Add Text Note"):
        st.text_input("Topic/Content", key="note_topic")
        st.slider("Importance Rating", 1, 10, key="note_rating")
        st.text_input("Tags (comma separated)", key="note_tags")
        st.button("Save Note", on_click=save_note)

    with st.expander("📂 Upload PDF"):
        unique_key = f"pdf_uploader_{st.session_state.uploader_key}"
        st.file_uploader("Choose a PDF", type="pdf", key=unique_key)
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
            # ADD HEADERS HERE
            res = requests.post(
                f"{API_URL}/ask", json=payload, headers=get_auth_headers()
            )

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
            elif res.status_code == 401:
                # If token expired, force logout
                st.error("Session expired. Please login again.")
                logout()
            else:
                message_placeholder.error(f"API Error: {res.text}")

        except Exception as e:
            message_placeholder.error(f"Connection Error: {e}")
