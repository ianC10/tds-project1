# Use an official Python runtime as a base image
FROM python:3.12

# Set the working directory inside the container
WORKDIR /app

# Copy only requirements.txt first for efficient caching
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application files
COPY . .

# Expose the port (if running a web server)
EXPOSE 8000

# Command to run the application
CMD ["python", "app/app.py"]
