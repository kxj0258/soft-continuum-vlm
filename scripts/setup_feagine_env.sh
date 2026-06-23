#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON:-python}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

if [[ -n "${FEAGINE_SIM_ROOT:-}" ]]; then
  FEAGINE_ROOT="${FEAGINE_SIM_ROOT}"
elif [[ -d "${PROJECT_ROOT}/../feagine_simulation" ]]; then
  FEAGINE_ROOT="${PROJECT_ROOT}/../feagine_simulation"
else
  FEAGINE_ROOT="${PROJECT_ROOT}/../feagine-simulation"
fi

echo "[INFO] Using Python: $(${PYTHON_BIN} -c 'import sys; print(sys.executable)')"
echo "[INFO] Activate your conda or venv environment before running this script."
echo "[INFO] Project root: ${PROJECT_ROOT}"
echo "[INFO] Feagine root: ${FEAGINE_ROOT}"

if [[ ! -d "${FEAGINE_ROOT}" ]]; then
  echo "[ERROR] Feagine root does not exist: ${FEAGINE_ROOT}"
  echo "[ERROR] Set FEAGINE_SIM_ROOT if your Feagine distribution uses another sibling path."
  exit 1
fi

if [[ -d "${FEAGINE_ROOT}/feagine-simulation-core" ]]; then
  CORE_DIR="${FEAGINE_ROOT}/feagine-simulation-core"
else
  CORE_DIR="${FEAGINE_ROOT}/dist/feagine-simulation-core"
fi
if [[ ! -d "${CORE_DIR}" ]]; then
  echo "[ERROR] Missing Feagine core directory: ${CORE_DIR}"
  exit 1
fi

echo "[INFO] Installing project Python dependencies into the current environment."
"${PYTHON_BIN}" -m pip install --upgrade pip
"${PYTHON_BIN}" -m pip install -e "${PROJECT_ROOT}"

echo "[INFO] Building and installing Feagine simulation core with CMake."
pushd "${CORE_DIR}" >/dev/null
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
cmake --install build --config Release
popd >/dev/null

echo "[INFO] Installing Feagine MuJoCo wheel."
MUJOCO_WHEEL="$(find "${FEAGINE_ROOT}" -maxdepth 3 -name 'feagine_mujoco-*.whl' | sort | tail -n 1 || true)"
if [[ -z "${MUJOCO_WHEEL}" ]]; then
  echo "[ERROR] Could not find feagine_mujoco-*.whl under ${FEAGINE_ROOT}"
  exit 1
fi
"${PYTHON_BIN}" -m pip install "${MUJOCO_WHEEL}"

SAPIEN_WHEEL="$(find "${FEAGINE_ROOT}" -maxdepth 3 -name 'feagine_sapien-*.whl' | sort | tail -n 1 || true)"
if [[ -n "${SAPIEN_WHEEL}" ]]; then
  if [[ "${INSTALL_FEAGINE_SAPIEN:-0}" == "1" ]]; then
    echo "[INFO] Installing optional Feagine SAPIEN wheel."
    "${PYTHON_BIN}" -m pip install "${SAPIEN_WHEEL}"
  else
    echo "[INFO] Optional SAPIEN wheel found but skipped: ${SAPIEN_WHEEL}"
    echo "[INFO] Re-run with INSTALL_FEAGINE_SAPIEN=1 to install it."
  fi
else
  echo "[INFO] No optional feagine_sapien-*.whl found."
fi

echo "[INFO] Feagine setup script finished. Run python scripts/verify_feagine_install.py next."
