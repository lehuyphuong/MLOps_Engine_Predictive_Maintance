FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY src/ ./src/

# Default entrypoint — override in CronJob spec with
#   args: ["python", "src/producer.py"]
# or
#   args: ["python", "src/validation_service.py"]
ENTRYPOINT ["python"]
CMD ["src/producer.py"]