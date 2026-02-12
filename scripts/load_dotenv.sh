#!/usr/bin/env bash
# Load environment variables from a .env file and export them

set -Eeuo pipefail

# You can pass the env file path as the first argument, or set ENV_FILE, or default to ".env"
ENV_FILE="${1:-${ENV_FILE:-.env}}"

if [[ ! -f "$ENV_FILE" ]]; then
  echo "Error: Env file not found: $ENV_FILE" >&2
    exit 1
fi

# Export all variable assignments found in the env file
# set -a enables 'allexport' so any variable assigned becomes exported automatically
set -a
# Strip potential Windows CRLFs while sourcing
# Process substitution requires bash; this script uses bash
source <(sed -e 's/\r$//' "$ENV_FILE")
set +a

# Example usage of loaded variables
: "${APP_ENV:=development}"
echo "Environment loaded from $ENV_FILE"
echo "APP_ENV=$APP_ENV"

# If an API token was provided, show its presence (without revealing it)
if [[ -n "${API_TOKEN:-}" ]]; then
  echo "API_TOKEN loaded (${#API_TOKEN} characters)"
fi

# Place your application startup command below, e.g.:
# exec ./your_app_binary
