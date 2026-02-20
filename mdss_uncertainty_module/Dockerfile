FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY pyproject.toml .
RUN pip install --upgrade pip && \
    pip install -e ".[dev]"

# Copy source
COPY src/ ./src/

# Create directories
RUN mkdir -p data/results logs

EXPOSE 8000

CMD ["uvicorn", "src.uncertainty_module.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
