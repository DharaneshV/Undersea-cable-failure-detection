"""Inspect the full TransformerEncoderBlock layer configs in the checkpoint."""
import zipfile, json

path = "checkpoints/best_model.keras"
with zipfile.ZipFile(path, "r") as z:
    config_str = z.read("config.json").decode()

config = json.loads(config_str)

def find_transformer_layers(obj, parent="", depth=0):
    if isinstance(obj, dict):
        cn = obj.get("class_name", "")
        cfg = obj.get("config", {})
        name = cfg.get("name", cn)
        
        if cn == "TransformerEncoderBlock":
            print(f"\n[TransformerEncoderBlock] name={name}")
            print(f"  config: num_heads={cfg.get('num_heads')}, ff_dim={cfg.get('ff_dim')}, dropout={cfg.get('dropout')}")
            
        if cn == "MultiHeadAttention":
            print(f"  {'  '*depth}MHA: num_heads={cfg.get('num_heads')}, key_dim={cfg.get('key_dim')}, name={name}")

        if cn == "Dense" and "dense" in name and depth < 8:
            print(f"  {'  '*depth}Dense: units={cfg.get('units')}, name={name}")
            
        for v in obj.values():
            find_transformer_layers(v, name, depth + 1)
    elif isinstance(obj, list):
        for item in obj:
            find_transformer_layers(item, parent, depth)

# Also check the raw build config
def find_build_configs(obj, depth=0):
    if isinstance(obj, dict):
        if obj.get("class_name") == "TransformerEncoderBlock":
            print(f"\nTransformerEncoderBlock build_config: {obj.get('build_config')}")
        for v in obj.values():
            find_build_configs(v, depth+1)
    elif isinstance(obj, list):
        for item in obj:
            find_build_configs(item, depth)

find_transformer_layers(config)
print("\n\n--- Build configs ---")
find_build_configs(config)
