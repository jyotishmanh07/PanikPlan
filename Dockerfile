# Use a lightweight Python image
FROM python:3.10-slim

# Install system dependencies and Poetry
RUN pip install poetry

# Set the working directory inside the container
WORKDIR /app

# Copy dependency files first (for better caching)
COPY pyproject.toml poetry.lock* ./

# Disable virtualenv creation and install project dependencies
RUN poetry config virtualenvs.create false && poetry install --no-root --no-interaction

# Copy the rest of your code and data
COPY . .

# Expose the port FastAPI runs on
EXPOSE 8000

# Start the application using uvicorn
CMD ["uvicorn", "src.stages.app:app", "--host", "0.0.0.0", "--port", "8000"]