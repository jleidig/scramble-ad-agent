# Use an official lightweight Python image as a base
FROM python:3.11.13-slim

# Set the working directory in the container
WORKDIR /app

# Setup environment
RUN pip install --upgrade pip && pip install pipenv
COPY Pipfile Pipfile.lock ./
RUN pipenv requirements > requirements.txt
RUN pip uninstall -y pipenv
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code into the container
COPY src ./src
RUN touch ./src/config/ia.ini

# Expose the port the FastAPI application runs on
EXPOSE 8080

# Command to run the application using uvicorn
CMD ["python", "-m", "src.main"]