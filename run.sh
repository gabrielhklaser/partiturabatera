#!/usr/bin/env bash
# DrumScribe — instala o que faltar e sobe o servidor local.
set -euo pipefail
cd "$(dirname "$0")"

PY=${PYTHON:-python3}
need=()
for m in numpy scipy soundfile miniaudio reportlab flask pypdfium2; do
  "$PY" -c "import $m" 2>/dev/null || need+=("$m")
done
if [ ${#need[@]} -gt 0 ]; then
  echo "instalando dependências: ${need[*]}"
  "$PY" -m pip install --quiet --disable-pip-version-check -r requirements.txt || {
    echo "o pip falhou — instale à mão: $PY -m pip install -r requirements.txt" >&2; exit 1; }
  faltam=()
  for m in "${need[@]}"; do "$PY" -c "import $m" 2>/dev/null || faltam+=("$m"); done
  if [ ${#faltam[@]} -gt 0 ]; then
    echo "continuam faltando: ${faltam[*]}" >&2
    echo "sem elas o servidor não sobe (import no topo de drumscribe/server.py)." >&2
    exit 1
  fi
fi

if [ ! -f samples/demo_drums.wav ]; then
  echo "gerando a faixa de demonstração…"
  "$PY" samples/make_demo.py >/dev/null
fi

PORT="${PORT:-8000}"
if command -v ss >/dev/null 2>&1 && ss -ltn 2>/dev/null | grep -q ":${PORT} "; then
  echo "a porta ${PORT} já está sendo usada — suba outro PORT=${PORT} ou encerre o processo." >&2
  exit 1
fi
echo "DrumScribe → http://localhost:${PORT}/   (Ctrl+C para sair)"
echo "diagnóstico:  curl http://localhost:${PORT}/api/health"
exec "$PY" -m drumscribe.server --host 0.0.0.0 --port "$PORT"
