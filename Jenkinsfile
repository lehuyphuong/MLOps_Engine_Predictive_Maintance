// Jenkinsfile — model-serving namespace CI/CD pipeline
//
// Stages:
//   1. Checkout           — clone repo from GitHub
//   2. Unit Test          — run test_serve.py, block deploy on failure (70% coverage gate)
//   3. Build Image        — docker build model-serving image
//   4. Push Image         — push to GCP Artifact Registry
//   5. Helm Dep Build     — build Helm chart dependencies
//   6. Deploy             — helm upgrade model-serving on GKE
//
// Requirements (Jenkins credentials):
//   gcp-sa-key  — GCP service account JSON key (Secret file)
//                 Used for both Artifact Registry auth and GKE kubeconfig
//                 (no separate kubeconfig file needed)

pipeline {

    agent any

    environment {
        GCP_PROJECT       = "aide2-494008"
        GCP_REGION        = "us-central1"
        GKE_CLUSTER       = "gke-phm"
        ARTIFACT_REGISTRY = "us-central1-docker.pkg.dev"
        IMAGE_REPO        = "us-central1-docker.pkg.dev/aide2-494008/phm-model-serving/model-serving"
        IMAGE_TAG         = "${BUILD_NUMBER}"
        GKE_NAMESPACE     = "model-serving"
        HELM_RELEASE      = "model-serving"
        HELM_CHART        = "charts/model-serving"
        TEST_DIR          = "services/model-serving/tests"
        SRC_DIR           = "services/model-serving/src"
        PYTHONUNBUFFERED  = "1"
    }

    options {
        timestamps()
        timeout(time: 20, unit: "MINUTES")
        buildDiscarder(logRotator(numToKeepStr: "10"))
        disableConcurrentBuilds()
    }

    stages {

        // ----------------------------------------------------------------
        // Stage 1 — Checkout
        // ----------------------------------------------------------------
        stage("Checkout") {
            steps {
                checkout scm
                echo "Commit: ${env.GIT_COMMIT?.take(8)}"
            }
        }

        // ----------------------------------------------------------------
        // Stage 2 — Unit Tests (gate — blocks deploy on failure)
        // ----------------------------------------------------------------
        stage("Unit Test") {
            steps {
                sh '''
                    set -e

                    python3 -m venv .venv
                    . .venv/bin/activate
                    pip install --quiet --upgrade pip
                    pip install --quiet \
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

                    python3 -m pytest ${TEST_DIR}/test_serve.py \
                        -v \
                        --tb=short \
                        --cov=${SRC_DIR} \
                        --cov-report=xml:coverage.xml \
                        --cov-report=term-missing \
                        --junit-xml=test-results.xml \
                        --cov-fail-under=70
                '''
            }
            post {
                always {
                    junit allowEmptyResults: true,
                        testResults: "test-results.xml"
                }
                failure {
                    echo "Unit tests FAILED — deployment blocked"
                }
            }
        }

        // ----------------------------------------------------------------
        // Stage 3 — Authenticate to GCP (single auth for all remaining stages)
        // ----------------------------------------------------------------
        stage("Authenticate to GCP") {
            steps {
                withCredentials([file(credentialsId: "gcp-sa-key",
                                      variable: "GCP_KEY")]) {
                    sh '''
                        set -e
                        gcloud auth activate-service-account \
                            --key-file="$GCP_KEY"
                        gcloud config set project ${GCP_PROJECT}
                        gcloud auth configure-docker \
                            ${ARTIFACT_REGISTRY} -q
                        gcloud container clusters get-credentials \
                            ${GKE_CLUSTER} \
                            --region ${GCP_REGION}
                    '''
                }
            }
        }

        // ----------------------------------------------------------------
        // Stage 4 — Build Docker Image
        // ----------------------------------------------------------------
        stage("Build Image") {
            steps {
                sh '''
                    set -e
                    cd services/model-serving
                    docker build \
                        -t ${IMAGE_REPO}:${IMAGE_TAG} \
                        -t ${IMAGE_REPO}:latest \
                        .
                '''
            }
        }

        // ----------------------------------------------------------------
        // Stage 5 — Push to Artifact Registry
        // ----------------------------------------------------------------
        stage("Push Image") {
            steps {
                sh '''
                    set -e
                    docker push ${IMAGE_REPO}:${IMAGE_TAG}
                    docker push ${IMAGE_REPO}:latest
                '''
            }
        }

        // ----------------------------------------------------------------
        // Stage 6 — Helm Dependency Build
        // ----------------------------------------------------------------
        stage("Helm Dependency Build") {
            steps {
                sh '''
                    set -e
                    helm dependency build ${HELM_CHART}
                '''
            }
        }

        // ----------------------------------------------------------------
        // Stage 7 — Deploy model-serving namespace via Helm
        // ----------------------------------------------------------------
        stage("Deploy") {
            steps {
                sh '''
                    set -e

                    kubectl cluster-info --request-timeout=10s

                    helm upgrade ${HELM_RELEASE} ${HELM_CHART} \
                        --namespace ${GKE_NAMESPACE} \
                        --create-namespace \
                        --atomic \
                        --timeout 5m \
                        --set deployment.image.tag=${IMAGE_TAG} \
                        --wait

                    kubectl rollout status \
                        deployment/${HELM_RELEASE} \
                        -n ${GKE_NAMESPACE} \
                        --timeout=300s
                '''
            }
            post {
                failure {
                    echo "Deployment FAILED — Helm --atomic ensures automatic rollback"
                }
            }
        }
    }

    post {
        success {
            echo "Pipeline SUCCESS — ${IMAGE_REPO}:${IMAGE_TAG} deployed to ${GKE_NAMESPACE}"
        }
        failure {
            echo "Pipeline FAILED at build ${BUILD_NUMBER}"
        }
        always {
            sh '''
                docker rmi ${IMAGE_REPO}:${IMAGE_TAG} 2>/dev/null || true
                docker rmi ${IMAGE_REPO}:latest 2>/dev/null || true
                rm -rf .venv 2>/dev/null || true
            '''
        }
    }
}