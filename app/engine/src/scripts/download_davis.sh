#!/usr/bin/env bash
# Download DAVIS 2017 (480p) into data_raw/DAVIS.
# DAVIS 2017 TrainVal 480p is small (~1-2 GB) and ships per-frame ground-truth masks --
# exactly what the davis_adapter needs (no SAM2 required for these).
# Run from the app/ directory:  bash scripts/download_davis.sh
set -euo pipefail

# Resolve target dir relative to this script: app/scripts/.. -> app/, then data_raw/
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEST="${SCRIPT_DIR}/../data_raw"
mkdir -p "${DEST}"
cd "${DEST}"

URL="https://data.vision.ee.ethz.ch/csergi/share/davis/DAVIS-2017-trainval-480p.zip"
ZIP="DAVIS-2017-trainval-480p.zip"

if [ -d "DAVIS/JPEGImages" ]; then
  echo "DAVIS already present at ${DEST}/DAVIS -- skipping download."
  exit 0
fi

echo "Downloading DAVIS 2017 (480p) to ${DEST} ..."
# curl with resume (-C -) so an interrupted download continues rather than restarts
if command -v curl >/dev/null 2>&1; then
  curl -L -C - -o "${ZIP}" "${URL}"
elif command -v wget >/dev/null 2>&1; then
  wget -c -O "${ZIP}" "${URL}"
else
  echo "ERROR: need curl or wget installed." >&2
  exit 1
fi

echo "Unzipping ..."
unzip -q -o "${ZIP}"
rm -f "${ZIP}"

echo "Done. Structure:"
echo "  ${DEST}/DAVIS/JPEGImages/480p/<clip>/00000.jpg ..."
echo "  ${DEST}/DAVIS/Annotations/480p/<clip>/00000.png ...  (ground-truth masks)"
echo ""
echo "Verify a clip exists:"
ls "${DEST}/DAVIS/JPEGImages/480p" 2>/dev/null | head -5 || echo "  (JPEGImages/480p not found -- check the unzip)"