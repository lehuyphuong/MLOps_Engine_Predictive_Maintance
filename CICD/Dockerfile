FROM jenkins/jenkins:lts-jdk21

USER root

# ----------------------------
# System dependencies
# ----------------------------
RUN apt-get update && apt-get install -y \
    curl \
    unzip \
    gnupg \
    ca-certificates \
    apt-transport-https \
    lsb-release \
    python3 \
    python3-venv \
    python3-pip \
    git \
    && rm -rf /var/lib/apt/lists/*

# ----------------------------
# Docker CLI
# ----------------------------
RUN curl -fsSL https://get.docker.com | sh

# ----------------------------
# kubectl
# ----------------------------
RUN curl -LO "https://dl.k8s.io/release/$(curl -Ls https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl" \
    && chmod +x kubectl \
    && mv kubectl /usr/local/bin/

# ----------------------------
# Helm 3
# ----------------------------
RUN curl https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3 | bash

# ----------------------------
# Google Cloud SDK + GKE auth plugin
# ----------------------------
RUN mkdir -p /usr/share/keyrings \
    && curl -fsSL https://packages.cloud.google.com/apt/doc/apt-key.gpg \
       | gpg --dearmor -o /usr/share/keyrings/cloud.google.gpg \
    && echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" \
       > /etc/apt/sources.list.d/google-cloud-sdk.list \
    && apt-get update \
    && apt-get install -y google-cloud-cli \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y \
    google-cloud-sdk-gke-gcloud-auth-plugin \
    && rm -rf /var/lib/apt/lists/*

# ----------------------------
# Python virtual environment — PEP 668 safe
# Includes all packages needed to run test_serve.py
# ----------------------------
RUN python3 -m venv /opt/venv \
    && /opt/venv/bin/pip install --no-cache-dir --upgrade pip \
    && /opt/venv/bin/pip install --no-cache-dir \
        pytest \
        pytest-cov \
        httpx \
        fastapi \
        uvicorn \
        pydantic \
        pandas \
        numpy \
        scikit-learn \
        xgboost \
        torch \
        redis \
        google-cloud-storage \
        prometheus-fastapi-instrumentator

ENV PATH="/opt/venv/bin:$PATH"

# ----------------------------
# Jenkins permissions
# ----------------------------
RUN usermod -aG docker jenkins

# ----------------------------
# Runtime declarations
# ----------------------------
VOLUME ["/var/jenkins_home"]

EXPOSE 8080 50000

USER jenkins
