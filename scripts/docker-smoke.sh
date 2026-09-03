#!/usr/bin/env bash
# Build the backend image, run it in demo mode, and check that it comes up
# healthy. Exits non-zero on any failure so CI can gate on it.
set -euo pipefail

IMAGE="kb-arena-docker-smoke:local"
CONTAINER="kb-arena-docker-smoke"
PORT="${DOCKER_SMOKE_PORT:-$(python3 -c 'import socket; s = socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')}"
DEADLINE_SECONDS=90

cleanup() {
  docker stop "$CONTAINER" >/dev/null 2>&1 || true
  docker rm "$CONTAINER" >/dev/null 2>&1 || true
}
trap cleanup EXIT

echo "Building $IMAGE from the repo Dockerfile..."
docker build -t "$IMAGE" .

echo "Starting $CONTAINER on port $PORT..."
docker run -d --name "$CONTAINER" -p "$PORT:8000" -e KB_ARENA_DEMO_MODE=true "$IMAGE" >/dev/null

echo "Polling http://localhost:$PORT/ready for up to ${DEADLINE_SECONDS}s..."
start_time=$(date +%s)
ready="false"
while true; do
  status=$(curl -s -o /dev/null -w "%{http_code}" "http://localhost:$PORT/ready" || echo "000")
  if [ "$status" = "200" ]; then
    ready="true"
    break
  fi
  elapsed=$(( $(date +%s) - start_time ))
  if [ "$elapsed" -ge "$DEADLINE_SECONDS" ]; then
    break
  fi
  sleep 2
done

if [ "$ready" != "true" ]; then
  echo "FAIL: /ready did not return HTTP 200 within ${DEADLINE_SECONDS}s"
  docker logs "$CONTAINER" || true
  exit 1
fi
echo "PASS: /ready returned HTTP 200"

health_body=$(curl -fsS "http://localhost:$PORT/health")
echo "Health body: $health_body"

health_status=$(python3 -c "import json, sys; print(json.loads(sys.argv[1])['status'])" "$health_body")
if [ "$health_status" != "ok" ]; then
  echo "FAIL: /health reported status '$health_status', expected 'ok'"
  exit 1
fi
echo "PASS: /health status is ok"

echo "Docker smoke test passed."
