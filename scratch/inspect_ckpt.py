import zipfile, json

path = "checkpoints/best_model.keras"
with zipfile.ZipFile(path, "r") as z:
    config_str = z.read("config.json").decode()

config = json.loads(config_str)

def find_layers(obj, depth=0):
    if isinstance(obj, dict):
        cn = obj.get("class_name", "")
        cfg = obj.get("config", {})
        if cn in ("Dense", "MultiHeadAttention", "Conv1D", "Conv1DTranspose"):
            size = cfg.get("units") or cfg.get("filters") or cfg.get("num_heads")
            name = cfg.get("name", cn)
            print(f"{'  '*depth}{cn:25s} name={name:30s} size={size}")
        for v in obj.values():
            find_layers(v, depth + 1)
    elif isinstance(obj, list):
        for item in obj:
            find_layers(item, depth)

find_layers(config)
