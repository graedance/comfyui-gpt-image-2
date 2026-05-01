# comfyui-gpt-image-2

ComfyUI custom nodes for OpenAI's [`gpt-image-2`](https://developers.openai.com/api/docs/models/gpt-image-2) (released 2026-04-21), with backwards compatibility for `gpt-image-1.5`, `gpt-image-1`, and `gpt-image-1-mini`.

## Nodes

- **GPT Image 2 — Generate** — text-to-image
- **GPT Image 2 — Edit** — image-to-image with up to **9 reference images** + optional mask
- **GPT Image 2 — Estimate Cost** — dry-run token + USD estimator (no API call)

All three live under the `OpenAI/GPT-Image-2` category in the right-click → Add Node menu.

## Install

### Option A — ComfyUI Manager
**Manager → Install via Git URL:**

```
https://github.com/graedance/comfyui-gpt-image-2
```

### Option B — git clone

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/graedance/comfyui-gpt-image-2.git
cd comfyui-gpt-image-2
pip install -r requirements.txt
```

For the **portable Windows** ComfyUI build, replace `pip` with:

```
..\..\python_embeded\python.exe -m pip install -r requirements.txt
```

Restart ComfyUI.

## API key

Set once as an environment variable (recommended) so you never paste it into a workflow:

```bash
# macOS / Linux
echo 'export OPENAI_API_KEY="sk-..."' >> ~/.zshrc
source ~/.zshrc
```

```powershell
# Windows
setx OPENAI_API_KEY "sk-..."
```

Or paste it into the optional `api_key` input on any node — useful for one-off runs but it'll be saved into the workflow JSON.

## Models

| Model ID | Notes |
|---|---|
| `gpt-image-2` | Default. Latest and recommended. |
| `gpt-image-2-2026-04-21` | Pinned snapshot for reproducibility. |
| `gpt-image-1.5` | Legacy. |
| `gpt-image-1` | Legacy. |
| `gpt-image-1-mini` | Cost-optimized legacy. |

## Sizes

`gpt-image-2` accepts any resolution within these constraints: max edge ≤ 3840 px, both edges multiples of 16, long:short ≤ 3:1, total pixels 655,360–8,294,400. Common values exposed in the dropdown:

`auto`, `1024x1024`, `1024x1536`, `1536x1024`, `1792x1024`, `1024x1792`, `2048x1152`, `2048x2048`, `3840x2160`, `2160x3840`

`gpt-image-1.x` is restricted to `1024x1024`, `1024x1536`, `1536x1024`, `auto`. Picking a 2K/4K size on a 1.x model will cause the API to reject the request.

## Cost estimator

The `Estimate Cost` node returns five outputs:

- `text_tokens` (INT)
- `image_in_tokens` (INT)
- `image_out_tokens` (INT)
- `usd_cost` (FLOAT)
- `summary` (STRING) — wire to a ShowText node for a human-readable breakdown

Example output for `4× high 1024² gpt-image-2`:

```
gpt-image-2 | 4× 1024x1024 high
  text in:       18 tok  → $0.0001
  image in:       0 tok  → $0.0000
  image out:  28132 tok  → $0.8440
  total:                  $0.8441
```

Pricing is hard-coded from OpenAI's published rates (text in $5/M, image in $8/M, cached $2/M, image out $30/M for `gpt-image-2`). Toggle `cached_input` to apply the cached image rate.

## License

MIT.
