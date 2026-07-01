#!/usr/bin/env bash
# docker-entrypoint.sh
# Chạy trước CMD mỗi khi container khởi động.
# Tạo symlink để infer_cli.py tìm được vocos tại ../checkpoints/vocos-mel-24khz
# (path hardcode trong infer_cli.py, tương đương /checkpoints/ từ WORKDIR=/app)
#
# F5TTS_CKPTS_DIR: cho phép ckpts nằm ở vị trí khác /app/ckpts — dùng trong image
# "deps-only" (Dockerfile.deps-only) nơi model được bake riêng ở /opt/f5tts/ckpts
# còn /app chỉ chứa code mount từ ngoài. Mặc định vẫn /app/ckpts như trước.
#
# GIT_REPO_URL (tùy chọn): nếu /app trống (chưa có gradio_app.py) và biến này được
# set, tự động git clone code vào /app khi container khởi động — dùng khi container
# có thể reach được Git remote (nội bộ/qua proxy công ty) dù không có internet chung.
# Nếu không set, container kỳ vọng /app đã được bind-mount sẵn code từ ngoài.

set -e

CKPTS_DIR="${F5TTS_CKPTS_DIR:-/app/ckpts}"

if [ -n "$GIT_REPO_URL" ] && [ ! -f "/app/gradio_app.py" ]; then
    echo "[entrypoint] /app chưa có code — cloning $GIT_REPO_URL (branch: ${GIT_REPO_BRANCH:-main}) ..."
    rm -rf /tmp/code-clone
    git clone --branch "${GIT_REPO_BRANCH:-main}" --depth 1 "$GIT_REPO_URL" /tmp/code-clone
    cp -a /tmp/code-clone/. /app/
    rm -rf /tmp/code-clone
    echo "[entrypoint] Đã clone code vào /app"
fi

# Vocos symlink: /checkpoints/vocos-mel-24khz → $CKPTS_DIR/vocos-mel-24khz
VOCOS_SRC="$CKPTS_DIR/vocos-mel-24khz"
VOCOS_LINK="/checkpoints/vocos-mel-24khz"

if [ -d "$VOCOS_SRC" ] && [ ! -L "$VOCOS_LINK" ]; then
    ln -sf "$VOCOS_SRC" "$VOCOS_LINK"
    echo "[entrypoint] Linked $VOCOS_LINK → $VOCOS_SRC"
elif [ ! -d "$VOCOS_SRC" ]; then
    echo "[entrypoint] WARNING: $VOCOS_SRC not found. CLI inference may fail."
    echo "             Mount ckpts/ volume hoặc set F5TTS_CKPTS_DIR trước khi khởi động container."
fi

exec "$@"
