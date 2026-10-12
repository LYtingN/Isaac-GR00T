#!/usr/bin/env bash

_GR00T_TACTILE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
_GR00T_TACTILE_VENV="${_GR00T_TACTILE_ROOT}/.venv"
_GR00T_FFMPEG_DIR="${_GR00T_TACTILE_VENV}/ffmpeg-runtime"

if [[ ! -f "${_GR00T_TACTILE_VENV}/bin/activate" ]]; then
    echo "Missing ${_GR00T_TACTILE_VENV}; run uv sync --python 3.12 from the repository root first." >&2
    return 1 2>/dev/null || exit 1
fi

# This workstation currently has a 570 kernel module but system-wide 580
# userspace symlinks. Prefer matching 570 libraries without changing /usr.
_GR00T_DRIVER_DIR="${_GR00T_TACTILE_VENV}/nvidia-driver-570"
if [[ -f /usr/lib/x86_64-linux-gnu/libcuda.so.570.172.18 ]]; then
    mkdir -p "${_GR00T_DRIVER_DIR}"
    ln -sfn /usr/lib/x86_64-linux-gnu/libcuda.so.570.172.18 "${_GR00T_DRIVER_DIR}/libcuda.so"
    ln -sfn /usr/lib/x86_64-linux-gnu/libcuda.so.570.172.18 "${_GR00T_DRIVER_DIR}/libcuda.so.1"
    ln -sfn /usr/lib/x86_64-linux-gnu/libnvidia-ml.so.570.172.18 "${_GR00T_DRIVER_DIR}/libnvidia-ml.so"
    ln -sfn /usr/lib/x86_64-linux-gnu/libnvidia-ml.so.570.172.18 "${_GR00T_DRIVER_DIR}/libnvidia-ml.so.1"
    export LD_LIBRARY_PATH="${_GR00T_DRIVER_DIR}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi

# TorchCodec needs FFmpeg's executable and shared libraries. Keep the runtime
# inside the project environment because this host does not provide FFmpeg.
if [[ -x "${_GR00T_FFMPEG_DIR}/bin/ffmpeg" ]]; then
    export PATH="${_GR00T_FFMPEG_DIR}/bin:${PATH}"
    export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:+${LD_LIBRARY_PATH}:}${_GR00T_FFMPEG_DIR}/lib"
fi

export NO_ALBUMENTATIONS_UPDATE=1
# shellcheck disable=SC1091
source "${_GR00T_TACTILE_VENV}/bin/activate"
unset _GR00T_DRIVER_DIR _GR00T_FFMPEG_DIR _GR00T_TACTILE_ROOT _GR00T_TACTILE_VENV
