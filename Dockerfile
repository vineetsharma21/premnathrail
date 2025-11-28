FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive

# Install system deps and a reasonable TeX Live subset for pdflatex
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       build-essential \
       ca-certificates \
       wget \
       fontconfig \
       fonts-dejavu-core \
       lmodern \
       texlive-latex-base \
       texlive-latex-recommended \
       texlive-latex-extra \
       texlive-fonts-recommended \
       texlive-science \
       texlive-pictures \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy app code
COPY . /app

# Create output folder and run as a non-root user
RUN addgroup --system appgroup \
    && adduser --system --ingroup appgroup appuser \
    && mkdir -p /app/output \
    && chown -R appuser:appgroup /app

USER appuser

EXPOSE 8000

# Use gunicorn with uvicorn workers for production
CMD ["gunicorn", "main:app", "-k", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:8000", "--timeout", "120"]