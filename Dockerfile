# 1. The Base Layer
# we start with an official Python image. "Slim" means it's stripped of junk.
FROM python:3.11-slim

# 2. Setup the work Directory
# This is like 'cd /app' inside the container.
WORKDIR /app

# 3. cache Dependencies (The Wizard's Trick)
# we copy ONLY requirements first. Why?
# Docker caches layers. If you change your code but not requirements,
# Docker skips installing packages again. It saves minutes on every build.
COPY requirements.txt .

# 4. Install Dependencies
# --no-cache-dir keeps the image small.
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy the Application Code
# Now we copy api.py, models.py, database.py into the container
COPY . .

# 6. make entry point executable
RUN chmod +x entrypoint.sh

# 7. The Launch Command
# This runs when the container starts.
CMD ["./entrypoint.sh"]

