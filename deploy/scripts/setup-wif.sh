#!/usr/bin/env bash
# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  setup-wif.sh — One-time GCP Workload Identity Federation setup         ║
# ║  para o pipeline GitHub Actions → GCP VM (NietzscheDB)                  ║
# ╚══════════════════════════════════════════════════════════════════════════╝
#
# Pré-requisitos:
#   - gcloud CLI instalado e autenticado como owner do projeto
#   - docker instalado na VM de destino
#   - curl instalado na VM de destino
#
# Uso:
#   export GCP_PROJECT_ID=seu-project-id
#   export GCP_REGION=us-central1
#   export GCP_VM_ZONE=us-central1-a
#   export GCP_VM_NAME=eva-mind-vm
#   export GITHUB_REPO=JoseRFJuniorLLMs/NietzscheDB
#   bash deploy/scripts/setup-wif.sh
#
# Após rodar, copie os valores de OUTPUT para os GitHub Secrets do repo.
# ──────────────────────────────────────────────────────────────────────────

set -euo pipefail

: "${GCP_PROJECT_ID:?Defina GCP_PROJECT_ID}"
: "${GCP_REGION:?Defina GCP_REGION}"
: "${GCP_VM_ZONE:?Defina GCP_VM_ZONE}"
: "${GCP_VM_NAME:?Defina GCP_VM_NAME}"
: "${GITHUB_REPO:?Defina GITHUB_REPO (ex: JoseRFJuniorLLMs/NietzscheDB)}"

GITHUB_ORG="${GITHUB_REPO%%/*}"
POOL_ID="github-pool"
PROVIDER_ID="github-provider"
SA_NAME="nietzsche-deploy"
AR_REPO="nietzsche"

PROJECT_NUMBER=$(gcloud projects describe "$GCP_PROJECT_ID" --format='value(projectNumber)')

echo ""
echo "══════════════════════════════════════════════"
echo "  Setup WIF — NietzscheDB Deploy"
echo "  Project: $GCP_PROJECT_ID ($PROJECT_NUMBER)"
echo "  Repo:    $GITHUB_REPO"
echo "══════════════════════════════════════════════"
echo ""

# ── 1. Habilitar APIs necessárias ─────────────────────────────────────────
echo "▶ Habilitando APIs..."
gcloud services enable \
  iam.googleapis.com \
  iamcredentials.googleapis.com \
  artifactregistry.googleapis.com \
  compute.googleapis.com \
  --project="$GCP_PROJECT_ID"

# ── 2. Criar Service Account ──────────────────────────────────────────────
echo "▶ Criando Service Account..."
gcloud iam service-accounts create "$SA_NAME" \
  --display-name="NietzscheDB GitHub Deploy" \
  --project="$GCP_PROJECT_ID" \
  2>/dev/null || echo "  (já existe — OK)"

SA_EMAIL="${SA_NAME}@${GCP_PROJECT_ID}.iam.gserviceaccount.com"

# ── 3. Criar Artifact Registry repo ──────────────────────────────────────
echo "▶ Criando Artifact Registry repo '$AR_REPO'..."
gcloud artifacts repositories create "$AR_REPO" \
  --repository-format=docker \
  --location="$GCP_REGION" \
  --description="NietzscheDB Docker images" \
  --project="$GCP_PROJECT_ID" \
  2>/dev/null || echo "  (já existe — OK)"

# ── 4. Permissões: SA → Artifact Registry ─────────────────────────────────
echo "▶ Concedendo permissão de push ao Artifact Registry..."
gcloud artifacts repositories add-iam-policy-binding "$AR_REPO" \
  --location="$GCP_REGION" \
  --member="serviceAccount:$SA_EMAIL" \
  --role="roles/artifactregistry.writer" \
  --project="$GCP_PROJECT_ID"

# ── 5. Permissões: SA → OS Login na VM ────────────────────────────────────
echo "▶ Concedendo OS Login na VM '$GCP_VM_NAME'..."
gcloud compute instances add-iam-policy-binding "$GCP_VM_NAME" \
  --zone="$GCP_VM_ZONE" \
  --member="serviceAccount:$SA_EMAIL" \
  --role="roles/compute.osAdminLogin" \
  --project="$GCP_PROJECT_ID"

# Permissão de viewer para SSH via IAP (se usar IAP)
gcloud projects add-iam-policy-binding "$GCP_PROJECT_ID" \
  --member="serviceAccount:$SA_EMAIL" \
  --role="roles/compute.viewer"

# ── 6. Criar Workload Identity Pool ───────────────────────────────────────
echo "▶ Criando Workload Identity Pool '$POOL_ID'..."
gcloud iam workload-identity-pools create "$POOL_ID" \
  --location=global \
  --display-name="GitHub Actions Pool" \
  --project="$GCP_PROJECT_ID" \
  2>/dev/null || echo "  (já existe — OK)"

# ── 7. Criar Provider no Pool ─────────────────────────────────────────────
echo "▶ Criando Provider '$PROVIDER_ID'..."
gcloud iam workload-identity-pools providers create-oidc "$PROVIDER_ID" \
  --location=global \
  --workload-identity-pool="$POOL_ID" \
  --display-name="GitHub OIDC Provider" \
  --issuer-uri="https://token.actions.githubusercontent.com" \
  --attribute-mapping="google.subject=assertion.sub,attribute.actor=assertion.actor,attribute.repository=assertion.repository,attribute.repository_owner=assertion.repository_owner" \
  --attribute-condition="assertion.repository_owner=='${GITHUB_ORG}'" \
  --project="$GCP_PROJECT_ID" \
  2>/dev/null || echo "  (já existe — OK)"

POOL_RESOURCE="projects/${PROJECT_NUMBER}/locations/global/workloadIdentityPools/${POOL_ID}/providers/${PROVIDER_ID}"

# ── 8. Binding: WIF → Service Account (apenas este repo) ──────────────────
echo "▶ Binding WIF → Service Account para repo '$GITHUB_REPO'..."
gcloud iam service-accounts add-iam-policy-binding "$SA_EMAIL" \
  --role="roles/iam.workloadIdentityUser" \
  --member="principalSet://iam.googleapis.com/projects/${PROJECT_NUMBER}/locations/global/workloadIdentityPools/${POOL_ID}/attribute.repository/${GITHUB_REPO}" \
  --project="$GCP_PROJECT_ID"

# ── 9. Habilitar OS Login na VM ───────────────────────────────────────────
echo "▶ Habilitando OS Login na VM..."
gcloud compute instances add-metadata "$GCP_VM_NAME" \
  --zone="$GCP_VM_ZONE" \
  --metadata=enable-oslogin=TRUE \
  --project="$GCP_PROJECT_ID"

# ── OUTPUT: GitHub Secrets ─────────────────────────────────────────────────
echo ""
echo "══════════════════════════════════════════════"
echo "  ✅ Setup concluído!"
echo "  Adicione estes valores como GitHub Secrets"
echo "  em: Settings → Secrets → Actions"
echo "══════════════════════════════════════════════"
echo ""
echo "GCP_PROJECT_ID     = $GCP_PROJECT_ID"
echo "GCP_REGION         = $GCP_REGION"
echo "GCP_ARTIFACT_REPO  = $AR_REPO"
echo "GCP_WIF_PROVIDER   = $POOL_RESOURCE"
echo "GCP_SERVICE_ACCOUNT= $SA_EMAIL"
echo "GCP_VM_NAME        = $GCP_VM_NAME"
echo "GCP_VM_ZONE        = $GCP_VM_ZONE"
echo "VM_COMPOSE_DIR     = /home/SEU_USUARIO/eva-mind   ← ajuste o path"
echo ""
echo "Imagem que será publicada:"
echo "  ${GCP_REGION}-docker.pkg.dev/${GCP_PROJECT_ID}/${AR_REPO}/nietzsche-server:latest"
echo ""
