# Use a lightweight Python image
FROM python:3.10-slim

# Set the working directory
WORKDIR /app

# Copy all your project files into the container
COPY . .

# Install the dependencies from your requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port Gradio usually runs on
EXPOSE 7860

# Command to run your app
CMD ["python", "app.py"]

