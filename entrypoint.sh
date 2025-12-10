#!/bin/bash

# 1. Start the Backend API in the background (&)
# We force it to run on port 8000 internally.
# The user (you) cannot access this port directly from the internet,
# but your Streamlit app (running on the same "machine") CAN access it at localhost:8000.
uvicorn api:app --host 0.0.0.0 --port 8000 &

# 2. Wait a few seconds for the API to wake up
sleep 5

# 3. Start the Frontend Streamlit
# We tell Streamlit to listen on the port Render assigns ($PORT).
# "address 0.0.0.0" allows the outside world to see it.
streamlit run app.py --server.port $PORT --server.address 0.0.0.0
