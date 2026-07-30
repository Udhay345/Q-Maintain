import requests
import json


class PhyphoxClient:
    """
    Phyphox remote HTTP client.

    Important API notes (phyphox wiki):
      /get?acc_x            -> last value only
      /get?acc_x=full       -> entire buffer contents
      /get?acc_x=42         -> values AFTER first sample exceeding threshold 42
                              (NOT an index / count — our old code was wrong)
    """

    def __init__(self, base_url):
        self.base_url = str(base_url).strip().rstrip("/")
        self.connected = False
        self.last_error = None
        self.last_status = None

    def _get_json(self, path, timeout=4.0):
        try:
            res = requests.get(f"{self.base_url}{path}", timeout=timeout)
            if res.status_code != 200:
                self.last_error = f"HTTP {res.status_code} for {path}"
                return None
            try:
                return res.json()
            except ValueError:
                self.last_error = f"Invalid JSON from {path}"
                return None
        except requests.exceptions.RequestException as e:
            self.last_error = str(e)
            return None

    def connect(self):
        """
        Verify reachability. Prefer /config (has buffer list), then /get, then /meta.
        Returns (success, response_json_or_error_message).
        """
        # /config lists buffers — best for mapping
        cfg = self._get_json("/config")
        if cfg is not None:
            self.connected = True
            return True, cfg

        data = self._get_json("/get")
        if data is not None:
            self.connected = True
            return True, data

        meta = self._get_json("/meta")
        if meta is not None:
            self.connected = True
            return True, meta

        self.connected = False
        return False, self.last_error or "Could not connect to Phyphox device"

    def start(self):
        data = self._get_json("/control?cmd=start")
        if data is None:
            return False
        # Phyphox returns {"result": true/false}
        if isinstance(data, dict) and "result" in data:
            return bool(data.get("result"))
        return True

    def stop(self):
        data = self._get_json("/control?cmd=stop")
        if data is None:
            return False
        if isinstance(data, dict) and "result" in data:
            return bool(data.get("result"))
        return True

    def list_buffer_names(self, payload):
        """Extract buffer names from /config, /get, or /meta JSON."""
        if not isinstance(payload, dict):
            return []

        names = []

        # /config format
        buffers = payload.get("buffers")
        if isinstance(buffers, list):
            for item in buffers:
                if isinstance(item, str):
                    names.append(item)
                elif isinstance(item, dict) and "name" in item:
                    names.append(item["name"])

        # /get format
        buffer_obj = payload.get("buffer")
        if isinstance(buffer_obj, dict):
            names.extend(buffer_obj.keys())

        # inputs[].outputs sometimes in /config
        inputs = payload.get("inputs")
        if isinstance(inputs, list):
            for inp in inputs:
                if not isinstance(inp, dict):
                    continue
                outputs = inp.get("outputs") or inp.get("output")
                if isinstance(outputs, dict):
                    names.extend(str(v) for v in outputs.values())
                elif isinstance(outputs, list):
                    for o in outputs:
                        if isinstance(o, str):
                            names.append(o)
                        elif isinstance(o, dict) and "buffer" in o:
                            names.append(o["buffer"])

        seen = set()
        ordered = []
        for n in names:
            if n and n not in seen:
                seen.add(n)
                ordered.append(n)
        return ordered

    def get_latest(self, buffer_names):
        """Return last value of each buffer: /get?acc_x&acc_y&..."""
        if not buffer_names:
            return self._get_json("/get")
        query = "&".join(str(b) for b in buffer_names)
        data = self._get_json(f"/get?{query}")
        if data and isinstance(data, dict):
            self.last_status = data.get("status")
        return data

    def get_full(self, buffer_names):
        """Return full contents of each buffer: /get?acc_x=full&acc_y=full&..."""
        if not buffer_names:
            return self._get_json("/get")
        query = "&".join(f"{b}=full" for b in buffer_names)
        data = self._get_json(f"/get?{query}")
        if data and isinstance(data, dict):
            self.last_status = data.get("status")
        return data

    def get_data(self, buffer_params=None):
        """
        Backward-compatible helper.
        Prefer get_full() / get_latest() for new code.
        """
        if buffer_params is None:
            return self._get_json("/get")
        if isinstance(buffer_params, dict) and buffer_params:
            # If values are the string "full" or literal full, request full buffers
            parts = []
            for k, v in buffer_params.items():
                if v == "full" or v is True:
                    parts.append(f"{k}=full")
                elif v == "latest" or v is None:
                    parts.append(str(k))
                else:
                    # Treat as threshold only if explicitly requested
                    parts.append(f"{k}={v}")
            return self._get_json("/get?" + "&".join(parts))
        if isinstance(buffer_params, list) and buffer_params:
            return self.get_latest(buffer_params)
        return self._get_json("/get")

    def is_measuring(self, payload=None):
        status = None
        if isinstance(payload, dict):
            status = payload.get("status")
        if status is None:
            status = self.last_status
        if isinstance(status, dict):
            return bool(status.get("measuring"))
        return None
