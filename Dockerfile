# Step A: Download a lightweight Linux environment with Python 3.11 pre-installed
FROM python:3.11-slim

# Step B: Set the working directory inside the container
WORKDIR /app

# Step C: Copy ONLY the requirements file first to leverage Docker layer caching
COPY requirements.txt .

# Step D: Install libraries (this heavy download is skipped on future builds if requirements.txt doesn't change)
RUN pip install --no-cache-dir -r requirements.txt

# Step E: Copy the actual API logic script
COPY main.py .

# Step F: Expose the network port
EXPOSE 8000

# Step G: The command to launch the FastAPI server using Uvicorn when the container starts
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]