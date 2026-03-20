#!/bin/bash
# POST JSON to NOTIFY_ENDPOINT. Exits 0 always.
# Usage: ./notify.sh "event_type" "message text"
[ -z "${NOTIFY_ENDPOINT:-}" ] && exit 0
AUTH_HEADER=""
[ -n "${NOTIFY_BEARER:-}" ] && AUTH_HEADER="Authorization: Bearer $NOTIFY_BEARER"
curl -s --connect-timeout 5 --max-time 10 \
  -X POST "$NOTIFY_ENDPOINT" \
  -H "Content-Type: application/json" \
  ${AUTH_HEADER:+-H "$AUTH_HEADER"} \
  -d "{\"event\":\"$1\",\"message\":\"$2\",\"timestamp\":\"$(date -u +%Y-%m-%dT%H:%M:%SZ)\",\"container\":\"single-autoquant\"}" \
  > /dev/null 2>&1 || true
exit 0
