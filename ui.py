import json
import requests
import streamlit as st

API_URL = "http://localhost:8000/ask"

st.set_page_config(page_title="RAG Assistant", layout="centered")
st.title("RAG Assistant")

query = st.chat_input("Ask something...")

if query:
    st.chat_message("user").write(query)

    with st.chat_message("assistant"):
        status = st.empty()
        answer_box = st.empty()

        status.status("⏳ Thinking...", state="running")

        answer_tokens = []
        try:
            with requests.post(API_URL, json={"query": query}, stream=True, timeout=120) as resp:
                resp.raise_for_status()
                for raw_line in resp.iter_lines():
                    if not raw_line:
                        continue
                    line = raw_line.decode("utf-8") if isinstance(raw_line, bytes) else raw_line
                    if not line.startswith("data:"):
                        continue
                    event = json.loads(line[5:].strip())

                    if event["type"] == "status":
                        status.status(event["message"], state="running")
                    elif event["type"] == "token":
                        if not answer_tokens:
                            status.empty()  # clear status once first token arrives
                        answer_tokens.append(event["content"])
                        answer_box.markdown("".join(answer_tokens))
                    elif event["type"] == "done":
                        status.empty()
                    elif event["type"] == "error":
                        status.empty()
                        st.error(event.get("message", "Unknown error"))

        except requests.exceptions.ConnectionError:
            status.empty()
            st.error("Cannot connect to API server. Make sure `api.py` is running on port 8000.")
        except Exception as e:
            status.empty()
            st.error(f"Error: {e}")
