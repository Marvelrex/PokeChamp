import os
import json
import time
import random
from collections import Counter
import requests

DEFAULT_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://192.168.149.1:11434")

class OllamaPlayer:
    def __init__(self, model: str = "gemma:7b", base_url: str = DEFAULT_BASE_URL, timeout: int = 180):
        self.model_id = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    # --- low-level helpers ---
    def _post(self, path: str, payload: dict):
        url = f"{self.base_url}{path}"
        r = requests.post(url, json=payload, timeout=self.timeout)
        r.raise_for_status()
        return r.json()

    def _generate(self, prompt, temperature, max_tokens,
                  seed=None, stop=None, stream=False, json_format=False):
        payload = {
            "model": self.model_id,
            "prompt": prompt,
            "stream": stream,
            "options": {
                "temperature": float(temperature),
                "num_predict": int(max_tokens),  # increase if outputs cut off
            },
        }
        if json_format:
            payload["format"] = "json"
        if seed is not None:
            payload["options"]["seed"] = int(seed)
        if stop:
            payload["options"]["stop"] = stop

        data = self._post("/api/generate", payload)
        return data.get("response", "")

    # --- public methods (API-compatible with LLAMAPlayer) ---

    def get_LLM_action(self, system_prompt, user_prompt, model=None, temperature=0.7,
                       json_format=True, seed=None, stop=None, max_tokens=200, actions=None):
        """
        Returns (message_or_json, is_json)
        """
        import json, re

        buffer = ("Hard limit: the entire output must be ≤ 200 tokens."
                  "Keep values concise (each string ≤ 20 words).\n\n")
        prompt = buffer + system_prompt + user_prompt
        text = self._generate(
            prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            seed=seed,
            stop=stop or [],
            stream=False,
            json_format=json_format
        )
        # 1) Try direct JSON
        try:
            obj = json.loads(text)
            return json.dumps(obj, ensure_ascii=False), True
        except Exception:
            print("Get LLM action failed!!!!!!!!!!!!!!!!!!")
            pass

        # 2) Try to grab first {...}
        s, e = text.find('{'), text.rfind('}')
        if s != -1 and e != -1 and e > s:
            candidate = text[s:e + 1]
            try:
                obj = json.loads(candidate)
                return candidate, True
            except Exception:
                pass

        # 3) Fallback: parse simple "**key:** value" lines
        thought = move = switch = None
        pat = re.compile(r'^\**\s*(thought|move|switch)\s*\**\s*:\s*(.+)$', re.I)
        for ln in text.splitlines():
            m = pat.match(ln.strip())
            if not m:
                continue
            k, v = m.group(1).lower(), m.group(2).strip().strip('` ')
            if k == "thought":
                thought = v
            elif k == "move":
                move = v
            elif k == "switch":
                switch = v
        if thought and (bool(move) ^ bool(switch)):
            obj = {"thought": thought}
            if move: obj["move"] = move
            if switch: obj["switch"] = switch
            return json.dumps(obj, ensure_ascii=False), True

        # 4) Otherwise return raw
        print("Gemma Output+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        print(text)
        return text, False
