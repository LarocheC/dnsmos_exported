#!/usr/bin/env bash
args=("$@")
for i in "${!args[@]}"; do
  if [[ "${args[$i]}" == "--json-quant-file" ]]; then
    q="${args[$((i+1))]}"
    python3 - "$q" <<'PY'
import json, sys
p = sys.argv[1]
d = json.load(open(p))
ts = d.get('tensors', {})
bools = [k for k, v in ts.items() if isinstance(v, dict) and v.get('format', {}).get('type') == 'BOOL']
for k in bools: del ts[k]
if bools: json.dump(d, open(p, 'w'), indent=1)
PY
  fi
done
exec "$(dirname "$0")/atonn.real" "$@"
