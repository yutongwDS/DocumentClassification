# Start with a base image
FROM python:3.9-slim

# Copy our application code
WORKDIR /app
COPY . .
COPY requirements.txt .

# Fetch app specific dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Expose port
EXPOSE 5000

# Start the app
CMD ["flask", "run", "--host=0.0.0.0", "--port=5000"]