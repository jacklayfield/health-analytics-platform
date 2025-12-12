#!/usr/bin/env bash
set -euo pipefail

###############################################
# CONFIG                                      #
###############################################

SYNTHEA_DIR="${SYNTHEA_DIR:-./synthea}"

OUTPUT_DIR="${OUTPUT_DIR:-./data/synthea}"

NUM_PATIENTS="${NUM_PATIENTS:-1000}"

# Optional region filters
STATE="${STATE:-}"
CITY="${CITY:-}"

###############################################
# Helper functions                             #
###############################################

log() { printf "\n\033[1;34m[run_synthea]\033[0m %s\n" "$1"; }

###############################################
# Clone synthea if missing                     #
###############################################
if [ ! -d "$SYNTHEA_DIR" ]; then
    log "Cloning Synthea..."
    git clone https://github.com/synthetichealth/synthea.git "$SYNTHEA_DIR"
fi

###############################################
# Build synthea                                #
###############################################
log "Building Synthea (this may take a minute)..."
(
    cd "$SYNTHEA_DIR"
    ./gradlew build -x test
)

###############################################
# Run Synthea                                   #
###############################################
log "Running Synthea..."

cd "$SYNTHEA_DIR"

CMD="./run_synthea -p $NUM_PATIENTS"

# Add filters if provided
if [ -n "$STATE" ]; then CMD="$CMD \"$STATE\""; fi
if [ -n "$CITY" ]; then CMD="$CMD \"$CITY\""; fi

# Execute
eval $CMD

###############################################
# Copy results into ETL folder                 #
###############################################
log "Copying output files..."

mkdir -p "$OUTPUT_DIR"

cp -r "$SYNTHEA_DIR/output/csv" "$OUTPUT_DIR/"
cp -r "$SYNTHEA_DIR/output/fhir" "$OUTPUT_DIR/"
cp -r "$SYNTHEA_DIR/output/ccda" "$OUTPUT_DIR/" 2>/dev/null || true

log "Done! CSVs are located at: $OUTPUT_DIR/csv"
