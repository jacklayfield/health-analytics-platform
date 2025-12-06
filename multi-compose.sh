#!/bin/bash

# Simple script to bring up or down multiple Docker Compose projects.

# Usage:
#   ./multi-compose.sh up
#   ./multi-compose.sh down

action="$1"   # "up" or "down"

if [[ "$action" != "up" && "$action" != "down" ]]; then
  echo "Error: must specify 'up' or 'down'"
  echo "Usage: $0 up|down"
  exit 1
fi

projects=(
  "infra"
  "etl"
  "ml"
  "dashboard"
)

for proj in "${projects[@]}"; do
  echo "Running '$action' for $proj..."

  (
    cd "$proj" || { echo "Could not cd into $proj"; exit 1; }

    if [[ "$action" == "up" ]]; then
      docker compose up -d
    else
      docker compose down
    fi
  )
done
