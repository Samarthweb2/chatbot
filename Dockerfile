# Dockerfile for Movie Recommender Chatbot
# Build: docker build -t movie-recommender:latest .
# Run: docker run -p 8000:8000 movie-recommender:latest

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Optionally install heavy ML deps at build time by passing --build-arg INSTALL_OPTIONAL=true
ARG INSTALL_OPTIONAL=false
RUN if [ "$INSTALL_OPTIONAL" = "true" ] ; then \
    pip install --no-cache-dir -r requirements-optional.txt || true ; \
    fi

COPY . /app

EXPOSE 8000
CMD ["uvicorn", "app.server:app", "--host", "0.0.0.0", "--port", "8000"]
