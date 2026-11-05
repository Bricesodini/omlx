"""
LFM Prefix Cache Test - Independent verification
"""
import json
import subprocess
import time
from pathlib import Path

PROD = "/Users/bot/Library/Application Support/oMLX-Prod"
api_key = Path(f"{PROD}/api.key").read_text().strip()

# Long system prompt to ensure prefix has enough tokens to cache
SYS = """Tu es un assistant expert en histoire médiévale. Réponds de manière structurée et pédagogique."""

def call_lfm(messages, max_tokens=100):
    """Call LFM and return (response, elapsed_ms)"""
    start = time.time()
    result = subprocess.run([
        "curl", "-s", "-m", "120",
        "-H", "Authorization: Bearer " + api_key,
        "-H", "Content-Type: application/json",
        "-d", json.dumps({
            "model": "lfm-2.6b-8bit",
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": 0.1,
        }),
        "http://127.0.0.1:8930/v1/chat/completions",
    ], capture_output=True, text=True)
    elapsed = (time.time() - start) * 1000
    return json.loads(result.stdout), elapsed

# Sequence of calls with growing prefix
print("=== LFM Cache Test ===\n")

# Call 1: short conversation
r1, t1 = call_lfm([
    {"role": "system", "content": SYS},
    {"role": "user", "content": "Cite 3 capitales européennes."},
])
u1 = r1.get("usage", {})
print(f"Call 1: prompt={u1.get('prompt_tokens')}, completion={u1.get('completion_tokens')}, cached={u1.get('cached_tokens', 0)}, elapsed={t1:.0f}ms")

# Call 2: same prefix + new question
r2, t2 = call_lfm([
    {"role": "system", "content": SYS},
    {"role": "user", "content": "Cite 3 capitales européennes."},
    {"role": "assistant", "content": r1["choices"][0]["message"]["content"]},
    {"role": "user", "content": "Et trois capitales asiatiques ?"},
])
u2 = r2.get("usage", {})
print(f"Call 2: prompt={u2.get('prompt_tokens')}, completion={u2.get('completion_tokens')}, cached={u2.get('cached_tokens', 0)}, elapsed={t2:.0f}ms")

# Call 3: longer prefix with same system
LONG_USR = "Décris en détail le rôle de la Hanse au Moyen Âge, ses routes commerciales, ses comptoirs principaux, et son influence politique en Europe du Nord. Cite au moins 5 dates." * 3
r3, t3 = call_lfm([
    {"role": "system", "content": SYS},
    {"role": "user", "content": LONG_USR},
])
u3 = r3.get("usage", {})
print(f"Call 3: prompt={u3.get('prompt_tokens')}, completion={u3.get('completion_tokens')}, cached={u3.get('cached_tokens', 0)}, elapsed={t3:.0f}ms")

# Call 4: same prefix + continuation
r4, t4 = call_lfm([
    {"role": "system", "content": SYS},
    {"role": "user", "content": LONG_USR},
    {"role": "assistant", "content": r3["choices"][0]["message"]["content"]},
    {"role": "user", "content": "Et quelle était la différence avec la Ligue hanséatique ?"},
])
u4 = r4.get("usage", {})
print(f"Call 4: prompt={u4.get('prompt_tokens')}, completion={u4.get('completion_tokens')}, cached={u4.get('cached_tokens', 0)}, elapsed={t4:.0f}ms")

print(f"\n=== Cache speedup ===")
print(f"Call 3 (cold): {t3:.0f}ms")
print(f"Call 4 (warm): {t4:.0f}ms")
if t3 > 0:
    print(f"Speedup: {t3/t4:.2f}x")
