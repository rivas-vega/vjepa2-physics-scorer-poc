
Deploy Meta's **V-JEPA 2** (Video Joint Embedding Predictive Architecture) with two complementary hallucination scoring metrics as a scalable GPU API on [Modal.com](https://modal.com).

**JEDi** (*JEPA Embedding Distance*) — [Luo et al., 2024](https://arxiv.org/abs/2410.05203) — is the state-of-the-art replacement for FVD for evaluating video generation quality:
- 34% more aligned with human perception than FVD
- Converges with only 16% of the samples FVD needs
- Sensitive to temporal distortions (I3D/FVD misses these)

**SurPrise** (*Violation of Expectation metric*) — [Garrido et al., 2025](https://arxiv.org/abs/2502.11831) — measures intuitive physics understanding via temporal coherence:
- No reference set needed — scores individual videos
- Two modes: single-video temporal coherence, or VoE pairwise accuracy
- Built on `hyle-surprice-metric-evaluator` using sliding-window V-JEPA 2 embeddings

---

## Project Structure

```
hyle_labs/
├── modal_app.py              # Modal deployment: GPU inference + FastAPI + middleware
├── jedi_scorer.py            ├── surprise_scorer.py        # SurPrise temporal coherence adapter (hyle-surprice-metric-evaluator)
├── preprocess.py             # Video preprocessing utilities (local testing)
├── client.py                 # Python client SDK
└── admin/
    ├── plans.py              # Billing plan definitions & limits
    ├── models.py             # SQLAlchemy ORM models
    ├── database.py           # SQLite engine on Modal Volume (/admin-db/)
    ├── auth.py               # API key + admin token authentication
    ├── billing.py            # Usage recording, prorated invoice generation
    ├── router.py                 ├── payment_router.py     # Redsys TPV payment endpoints
    ├── redsys.py             # Redsys HMAC_SHA256_V1 crypto helpers
    └── dashboard.html        # Single-page admin UI
```

---

## Quick Start

### 1. Install & authenticate Modal

```bash
pip install modal requests
modal setup
```

### 2. Configure secrets

```bash
modal secret create hyle-admin-secret HYLE_ADMIN_SECRET=my-secure-token

# Redsys payment integration (optional — leave unset to stay in demo/sandbox mode)
modal secret create hyle-redsys \
  REDSYS_MERCHANT_CODE=999008881 \
  REDSYS_TERMINAL=001 \
  REDSYS_SECRET=sq7HjrUOBfKmC576ILgskD5srU870gJ7 \
  REDSYS_DEMO=1 \
  HYLE_BASE_URL=https://<your-modal-url>
```

### 3. Deploy

```bash
modal deploy modal_app.py
# → https://your-org--vjepa2-inference-api-server.modal.run
```

### 4. Download model weights (once)

```bash
curl -X POST https://<your-url>/download-model \
  -H "Content-Type: application/json" \
  -d '{"model_variant": "vjepa2-vitl-fpc64-256"}'
```

### 5. Run inference

```bash
# Full pipeline (upload → preprocess → encode)
curl -X POST https://<your-url>/pipeline \
  -H "X-API-Key: hyl_yourkey" \
  -F "file=@my_video.mp4"
```

---

## Authentication

| Header | Required on | Value |
|--------|-------------|-------|
| `X-API-Key` | All API endpoints | `hyl_...` — create via admin dashboard |
| `X-Admin-Token` | `/admin/*` endpoints | Token set in `hyle-admin-secret` Modal secret |

```bash
# API key usage
curl -X POST https://<your-url>/pipeline \
  -H "X-API-Key: hyl_yourkey" \
  -F "file=@video.mp4"

curl https://<your-url>/admin/stats \
  -H "X-Admin-Token: my-secure-token"
```

Common error responses:

```json
{"detail": "Missing X-API-Key header."}    // 401
{"detail": "Invalid or revoked API key."}  // 401

// 429 — monthly quota exhausted (Free plan)
{
  "detail": "Monthly limit reached: 10 min included in the Free Tier plan.",
  "plan_id": "free",
  "included_minutes": 10,
  "used_minutes": 10.312
}

// 413 — file too large for plan
{
  "detail": "File too large: 210.4 MB exceeds the 100 MB limit for the Free Tier plan.",
  "plan_id": "free",
  "max_video_size_mb": 100,
  "file_size_mb": 210.4
}

// 429 — video storage limit reached
{
  "detail": "Storage limit reached: 20 videos allowed on the Free Tier plan.",
  "plan_id": "free",
  "max_videos_stored": 20,
  "current_stored": 20
}
```

---

## API Reference

### Setup

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET`  | `/health` | GPU & model health check |
| `POST` | `/download-model` | Download V-JEPA 2 weights to persistent volume |

### Video Management

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST`   | `/upload` | Upload a video file → `video_id` |
| `POST`   | `/preprocess` | Preprocess an uploaded video |
| `GET`    | `/videos` | List all stored videos |
| `DELETE` | `/videos/{id}` | Delete a video |

#### `POST /upload` — plan limits enforced at upload time

```bash
curl -X POST https://<your-url>/upload \
  -H "X-API-Key: hyl_..." \
  -F "file=@video.mp4"
# → {"video_id": "3f8a1c..."}
```

Limits checked per plan:
- **Max file size** (`max_video_size_mb`) — returns `413` if exceeded
- **Max stored videos** (`max_videos_stored`) — returns `429` if at capacity

#### `POST /preprocess`

```json
{
  "target_fps": 8,
  "num_frames": 16,
  "frame_size": 224,
  "sampling_mode": "uniform",
  "stride": 2,
  "normalize_mean": [0.485, 0.456, 0.406],
  "normalize_std":  [0.229, 0.224, 0.225]
}
```

### Inference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/encode` | Extract V-JEPA 2 embeddings |
| `POST` | `/pipeline` | upload → preprocess → encode in one call |

```bash
curl -X POST /pipeline?task=encode \
  -H "X-API-Key: hyl_..." \
  -F "file=@video.mp4" \
  -F 'config_json={"target_fps":8,"num_frames":16,"frame_size":224}'
```

**GPU routing by plan:** inference endpoints automatically route to the appropriate GPU tier based on the user's plan — no extra configuration needed.

| GPU | Plans | Hardware |
|-----|-------|----------|
| Standard | Free, Pay-per-Minute, Starter | A10G |
| Premium | Professional, Enterprise | A100 40GB |


| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/jedi/score` | Score generated vs reference videos |
| `POST` | `/jedi/corpus` | Build & store a reference corpus |
| `GET`  | `/jedi/corpus` | List stored corpora |
| `POST` | `/jedi/score-corpus` | Score against pre-stored corpus |
| `POST` | `/jedi/pipeline` | upload → preprocess → encode → JEDi in one call |

```bash
curl -X POST "/jedi/pipeline?reference_video_ids=id1,id2,id3&kernel=polynomial" \
  -H "X-API-Key: hyl_..." \
  -F "file=@generated.mp4"
```

### SurPrise — Hallucination Scoring

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/surprise/score` | Score videos (single-video or VoE pair mode) |
| `POST` | `/surprise/pipeline` | upload → preprocess → encode → SurPrise in one call |

**Single-video mode** — measures temporal coherence of one video (no reference set needed):

```bash
curl -X POST https://<your-url>/surprise/score \
  -H "X-API-Key: hyl_..." \
  -H "Content-Type: application/json" \
  -d '{
    "video_ids": ["<preprocessed_video_id>"],
    "distance_metric": "l1",
    "aggregation_mode": "average",
    "window_size": 8,
    "context_length": 4,
    "stride": 2
  }'
# → {
#     "mode": "single_video",
#     "videos": [{
#       "video_id": "...",
#       "surprise_score": 0.083,
#       "max_surprise": 0.142,
#       "hallucination_level": "low",
#       "hallucination_score": 38.2,
#       "n_windows": 12,
#       "window_scores": [0.071, 0.089, ...]
#     }]
#   }
```

**VoE pair mode** — evaluates whether the model is more surprised by physically impossible videos:

```bash
curl -X POST https://<your-url>/surprise/score \
  -H "X-API-Key: hyl_..." \
  -H "Content-Type: application/json" \
  -d '{
    "possible_video_ids":   ["vid_possible_1", "vid_possible_2"],
    "impossible_video_ids": ["vid_impossible_1", "vid_impossible_2"],
    "distance_metric": "l1"
  }'
# → {
#     "mode": "voe_pair",
#     "pairwise_accuracy": 0.75,
#     "n_correct_pairs": 3,
#     "n_total_pairs": 4,
#     "possible_scores":   [0.071, 0.088],
#     "impossible_scores": [0.143, 0.062]
#   }
```

**Full pipeline (upload → score in one call):**

```bash
curl -X POST "https://<your-url>/surprise/pipeline?distance_metric=l1&window_size=8&context_length=4" \
  -H "X-API-Key: hyl_..." \
  -F "file=@video.mp4"
```


| | JEDi | SurPrise |
|---|---|---|
| **What it measures** | Distributional distance between two sets of videos | Temporal coherence within individual videos |
| **Reference set needed** | Yes (≥ 2 reference videos) | No |
| **Use case** | Evaluate a generative model against a real dataset | Detect temporally incoherent or physically implausible videos |
| **Metric** | MMD² in V-JEPA 2 embedding space | Mean L1 window surprise |
| **Paper** | [arXiv:2410.05203](https://arxiv.org/abs/2410.05203) | [arXiv:2502.11831](https://arxiv.org/abs/2502.11831) |


All `/admin/*` routes require `X-Admin-Token`. The Swagger UI at `/docs` has an **Authorize** button for this.

#### Users

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET`    | `/admin/stats` | Platform-wide aggregate stats |
| `GET`    | `/admin/users` | List users (search, plan filter, pagination) |
| `POST`   | `/admin/users` | Create a user |
| `GET`    | `/admin/users/{id}` | User detail + usage summary + plan history |
| `PUT`    | `/admin/users/{id}` | Update user (name, plan, active status) |
| `DELETE` | `/admin/users/{id}` | Deactivate user + revoke all keys |

```bash
# Create a user
curl -X POST https://<your-url>/admin/users \
  -H "X-Admin-Token: my-token" \
  -H "Content-Type: application/json" \
  -d '{"email": "alice@example.com", "name": "Alice", "plan_id": "starter"}'

# Upgrade plan (recorded in plan history for prorated invoicing)
curl -X PUT https://<your-url>/admin/users/<user_id> \
  -H "X-Admin-Token: my-token" \
  -H "Content-Type: application/json" \
  -d '{"plan_id": "professional"}'
```

Changing a user's plan writes a `PlanHistory` record. Invoice generation automatically uses this history to compute prorated charges (see [Billing](#billing--invoicing)).

#### API Keys

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET`    | `/admin/api-keys` | List active keys (filter by `user_id`) |
| `POST`   | `/admin/api-keys` | Create API key for a user |
| `DELETE` | `/admin/api-keys/{id}` | Revoke a key |

```bash
# Create an API key — the raw key is returned only once
curl -X POST https://<your-url>/admin/api-keys \
  -H "X-Admin-Token: my-token" \
  -H "Content-Type: application/json" \
  -d '{"user_id": "<uuid>", "name": "Production"}'
# → {"key": "hyl_abc123...", "key_prefix": "hyl_abc", ...}
```

#### Usage

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET`  | `/admin/usage` | Query usage records (filter by user, date range) |
| `GET`  | `/admin/usage/summary` | Aggregated daily breakdown (last N days) |
| `POST` | `/admin/usage/record` | Record a usage event (internal) |

```bash
# Usage for a specific user in May
curl "https://<your-url>/admin/usage?user_id=<uuid>&since=2025-05-01T00:00:00&until=2025-05-31T23:59:59" \
  -H "X-Admin-Token: my-token"
```

#### Billing & Invoicing

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET`  | `/admin/plans` | List all plans with limits |
| `GET`  | `/admin/invoices` | List invoices (filter by user, status) |
| `POST` | `/admin/invoices/generate` | Generate a prorated invoice |
| `GET`  | `/admin/invoices/{id}` | Invoice detail with line items |
| `PUT`  | `/admin/invoices/{id}/status` | Mark as `paid` / `void` |

**Prorated invoicing** — if a user changed plans mid-period, the invoice is split into segments:

```bash
curl -X POST https://<your-url>/admin/invoices/generate \
  -H "X-Admin-Token: my-token" \
  -H "Content-Type: application/json" \
  -d '{"user_id": "<uuid>", "period_start": "2025-05-01T00:00:00", "period_end": "2025-05-31T23:59:59"}'
```

Example line items for a user who upgraded from Starter → Professional on May 15:

```json
{
  "amount_usd": 202.58,
  "line_items": [
    {"description": "Starter subscription (14d, 45%)",    "quantity": 0.4516, "unit_price": 99,  "total": 44.71},
    {"description": "Professional subscription (17d, 55%)","quantity": 0.5484, "unit_price": 299, "total": 163.97},
    {"description": "Overage (14d, 45%) — 38.2 min @ $0.04/min", "quantity": 38.2, "unit_price": 0.04, "total": 1.53}
  ]
}
```

If no plan change occurred during the period, a single line item is generated (same as before).

### Payments — Redsys TPV

| Method | Endpoint | Auth | Description |
|--------|----------|------|-------------|
| `POST` | `/admin/payments/{user_id}/checkout` | Admin | Create a payment order, get `payment_url` |
| `GET`  | `/admin/payments` | Admin | List payment orders |
| `GET`  | `/admin/payments/{id}/status` | Admin | Get order status |
| `GET`  | `/payments/{order_id}` | Public | Redsys auto-submit form (share with user) |
| `POST` | `/payments/notify` | Public | Redsys S2S callback (set as `DS_MERCHANT_MERCHANTURL`) |
| `GET`  | `/payments/ok` | Public | Success landing page |
| `GET`  | `/payments/ko` | Public | Error landing page |

#### Payment flow

```
Admin dashboard
  └─► POST /admin/payments/{user_id}/checkout
        └─► returns { payment_url, redsys_order, amount_usd, ... }

Share payment_url with user
  └─► User opens /payments/{order_id}
        └─► HTML page auto-POSTs signed form to Redsys sandbox/prod
              └─► User completes payment on Redsys hosted page
                    ├─► Redsys POSTs to /payments/notify (signature verified)
                    │     └─► Order status → "paid" | "failed"
                    └─► User redirected to /payments/ok or /payments/ko
```

```bash
# Create a payment order for the user's current plan
curl -X POST https://<your-url>/admin/payments/<user_id>/checkout \
  -H "X-Admin-Token: my-token" \
  -H "Content-Type: application/json" \
  -d '{}'
# → {
#     "id": "...",
#     "redsys_order": "56781A2B",
#     "amount_usd": 99.0,
#     "status": "pending",
#     "payment_url": "https://<your-url>/payments/<order_id>"
#   }

# Override plan (e.g. let user pay for an upgrade before switching)
curl -X POST https://<your-url>/admin/payments/<user_id>/checkout \
  -H "X-Admin-Token: my-token" \
  -H "Content-Type: application/json" \
  -d '{"plan_id": "professional"}'

# List all payment orders
curl "https://<your-url>/admin/payments?status=pending" \
  -H "X-Admin-Token: my-token"
```

#### Redsys environment variables

| Variable | Default (sandbox) | Description |
|----------|-------------------|-------------|
| `REDSYS_MERCHANT_CODE` | `999008881` | Merchant code from Redsys |
| `REDSYS_TERMINAL` | `001` | Terminal number |
| `REDSYS_SECRET` | demo key | HMAC secret from Redsys control panel |
| `REDSYS_DEMO` | `1` | Set to `0` to switch to production URL |
| `HYLE_BASE_URL` | Modal default URL | Base URL for notify/ok/ko callbacks |

All default to Redsys sandbox (`sis-t.redsys.es:25443`). Set `REDSYS_DEMO=0` and provide real credentials via Modal secrets to go live.

---


Open `https://<your-url>/admin/dashboard` in a browser and enter your admin token.

### Sections

| Section | Description |
|---------|-------------|
| **Overview** | Live stats (active users, API keys, minutes MTD, estimated revenue), 30-day usage bar chart, plan distribution doughnut |
| **Users** | Searchable/filterable table; per-row actions: View, + Key, Invoice, Disable |
| **API Keys** | All active keys with user name, last-used timestamp; revoke button |
| **Usage** | Usage records filtered by user (dropdown) and date range |
| **Invoices** | All invoices with user name; mark paid / void |
| **Payments** | Redsys payment orders with status badges |
| **Plans** | Plan cards with all limits and pricing |

### Creating a user and issuing an API key

1. **Users → + New User** — fill email, name, plan → Create
2. Click **+ Key** on the user's row (or **API Keys → + New Key** with the user pre-selected)
3. Copy the key — it is shown only once

### Generating a payment link

1. Open a user via **View** in the Users table
2. Optionally change the plan in the selector
3. Click **Pay Plan** — a new tab opens with the Redsys payment page
4. Share the URL with the user; the tab auto-redirects to Redsys checkout

### Generating a prorated invoice

1. **Invoices → Generate Invoice** (or click **Invoice** on a user row)
2. Select the user, set the billing period, click Generate
3. If the user changed plans during the period, the invoice automatically splits into prorated segments

---

## Plans

| Plan | Monthly Fee | Included Min | Overage | Max File | Max Videos | GPU |
|------|------------|-------------|---------|----------|------------|-----|
| Free | $0 | 10 | blocked | 100 MB | 20 | Standard (A10G) |
| Pay-per-Minute | $0 | — | $0.05/min | 500 MB | 500 | Standard (A10G) |
| Starter | $99 | 500 | $0.04/min | 500 MB | 2,000 | Standard (A10G) |
| Professional | $299 | 2,000 | $0.03/min | 500 MB | 10,000 | **Premium (A100)** |
| Enterprise | $999 | Unlimited | — | 500 MB | 100,000 | **Premium (A100)** |

**Quota enforcement at runtime:**

| Plan | Behaviour when limit reached |
|------|------------------------------|
| Free | `429` hard block — no overage |
| Pay-per-Minute | Continues; billed per minute |
| Starter / Professional | Continues; overage billed at plan rate |
| Enterprise | Never blocked |

---

## Score Interpretation


| MMD² Score | Level | Meaning |
|-----------|-------|---------|
| < 0.001 | **None** | Indistinguishable from real video distribution |
| 0.001–0.005 | **Low** | Minor temporal or spatial artifacts |
| 0.005–0.02 | **Moderate** | Noticeable hallucinations likely present |
| 0.02–0.08 | **High** | Significant hallucinations or unrealistic content |
| > 0.08 | **Severe** | Far outside real distribution |

### SurPrise

| Avg L1 Surprise | Level | Meaning |
|----------------|-------|---------|
| < 0.05 | **None** | Temporally coherent — no hallucination proxy signal |
| 0.05–0.15 | **Low** | Slightly elevated — minor artifacts or frame drift |
| 0.15–0.35 | **Moderate** | Noticeable temporal inconsistency |
| 0.35–0.70 | **High** | Strong temporal surprise — likely hallucinations |
| > 0.70 | **Severe** | Extreme temporal incoherence |

Thresholds are calibrated for L1 distance on V-JEPA 2 normalised embeddings. The `aggregation_mode=maximum` setting uses the worst-case window score, which is more sensitive for detecting isolated temporal artifacts.

---

## Python Client

```python
from client import VJepa2Client

client = VJepa2Client("https://<your-url>", api_key="hyl_yourkey")

# Health check
print(client.health())

# Full pipeline — upload → preprocess → encode
result = client.pipeline("video.mp4", num_frames=16, frame_size=224)
embeddings = result["encode"]["embeddings"]
print(f"Embedding dim: {len(embeddings[0])}")

# Step by step
upload     = client.upload("video.mp4")
preprocess = client.preprocess(upload["video_id"], num_frames=64, frame_size=256)
encode     = client.encode(upload["video_id"])
```

### SurPrise Workflows

**Single-video temporal coherence:**

```python
vid_id = client.upload_and_preprocess("generated_video.mp4")

result = client.surprise_score([vid_id])
video  = result["videos"][0]
print(f"Surprise: {video['surprise_score']:.4f}")
print(f"Level:    {video['hallucination_level']}")     # "low"
print(f"Score:    {video['hallucination_score']:.1f}/100")
print(f"Windows:  {video['n_windows']}")
```

**VoE pairwise accuracy:**

```python
possible_ids   = [client.upload_and_preprocess(p) for p in possible_videos]
impossible_ids = [client.upload_and_preprocess(p) for p in impossible_videos]

result = client.surprise_score_voe(possible_ids, impossible_ids)
print(f"VoE accuracy: {result['pairwise_accuracy']:.2%}")   # e.g. "75.00%"
print(f"Correct pairs: {result['n_correct_pairs']} / {result['n_total_pairs']}")
```

**Full pipeline — upload → preprocess → encode → score:**

```python
result   = client.surprise_pipeline("video.mp4")
video    = result["surprise"]["videos"][0]
print(f"Level: {video['hallucination_level']}")
```


**A — Ad-hoc scoring**

```python
ref_ids = [client.upload_and_preprocess(p) for p in real_video_paths]
gen_ids = [client.upload_and_preprocess(p) for p in generated_paths]

result = client.jedi_score(gen_ids, ref_ids)
print(result["hallucination_level"])   # "low"
print(result["jedi_score"])            # 0.00312
print(result["hallucination_score"])   # 28.4
```

**B — Persistent corpus (recommended for repeated evaluation)**

```python
# Build once
ref_ids = [client.upload_and_preprocess(p) for p in real_video_paths]
client.jedi_build_corpus(ref_ids, corpus_name="ucf101_reference")

# Score many videos against the stored corpus
for gen_path in new_generated_videos:
    gen_id = client.upload_and_preprocess(gen_path)
    result = client.jedi_score_corpus([gen_id], corpus_name="ucf101_reference")
    print(f"{gen_path}: {result['hallucination_level']} (JEDi={result['jedi_score']:.4f})")
```

**C — Full pipeline in one call**

```python
result = client.jedi_pipeline(
    "generated_video.mp4",
    reference_video_ids=ref_ids,
    num_frames=16,
    frame_size=224,
)
print(result["jedi"]["interpretation"])
```

**D — Batch evaluation**

```python
result = client.evaluate_batch(
    generated_paths=["gen1.mp4", "gen2.mp4", "gen3.mp4"],
    reference_paths=["real1.mp4", "real2.mp4", "real3.mp4"],
    build_corpus=True,
    corpus_name="my_reference",
)
```

---

## CLI Usage

```bash
# Pass the API key via --key or the HYLE_API_KEY environment variable
export HYLE_API_KEY=hyl_yourkey

python client.py <url> upload real1.mp4
python client.py <url> preprocess <ref_id>
python client.py <url> encode <video_id>
python client.py <url> pipeline video.mp4
python client.py <url> videos
python client.py <url> delete <video_id>

python client.py <url> jedi-pipeline generated.mp4 ref_id1,ref_id2,ref_id3
python client.py <url> jedi-corpus ref_id1,ref_id2,ref_id3 my_corpus
python client.py <url> jedi-score-corpus gen_id1,gen_id2 my_corpus
python client.py <url> jedi-score gen_id1,gen_id2 ref_id1,ref_id2,ref_id3

# SurPrise — single-video temporal coherence
python client.py <url> surprise-pipeline video.mp4
python client.py <url> surprise-score vid_id1,vid_id2

# SurPrise — VoE pairwise accuracy
python client.py <url> surprise-score-voe possible_id1,possible_id2 impossible_id1,impossible_id2
```

---

## Architecture

```
User Request
     │
     ▼
FastAPI (Modal CPU, 2 cores)
  ├── AuthAndBillingMiddleware
  │     ├── 1. Validate X-API-Key                (→ 401 if missing/invalid)
  │     ├── 2. Check plan limits at upload time  (→ 413 / 429)
  │     ├── 3. Quota check for hard-cap plans    (→ 429 if exhausted)
  │     ├── 4. Stamp user_id, api_key_id, priority_gpu on request.state
  │     ├── 5. Forward to endpoint
  │     └── 6. Record UsageRecord (video_minutes, cost_usd)
  ├── /upload            ──► Modal Volume (video_volume) + owner.json sidecar
  ├── /preprocess        ──► CPU Worker → (C,T,H,W) tensor
  ├── /encode            ──► VJepa2ModelStandard or VJepa2ModelPremium
  ├── /jedi/*            ──► VJepa2ModelStandard or VJepa2ModelPremium
  ├── /surprise/*        ──► VJepa2ModelStandard or VJepa2ModelPremium
  ├── /admin/*           ──► Admin router (X-Admin-Token)
  └── /payments/*        ──► Payment router (public + admin)

GPU Tier Selection (per request, based on plan):
  ┌─────────────────────────────────────────────┐
  │ priority_gpu = False → VJepa2ModelStandard  │  A10G
  │ priority_gpu = True  → VJepa2ModelPremium   │  A100 40GB
  └─────────────────────────────────────────────┘

Modal Volumes:
  ├── vjepa2-weights      (model checkpoints)
  ├── vjepa2-videos       (uploaded videos + owner.json)
  ├── vjepa2-jedi-store   (reference corpora)
  └── hyle-admin-db       (SQLite: users, api_keys, usage_records,
                                    invoices, plan_history, payment_orders)
```

---

## Preprocessing Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `target_fps` | `8` | FPS for frame extraction |
| `num_frames` | `16` | Frames to sample per clip |
| `frame_size` | `224` | Output square frame size (px) |
| `sampling_mode` | `uniform` | `uniform` / `random` / `strided` |
| `stride` | `2` | Step size for strided sampling |
| `normalize_mean` | `[0.485, 0.456, 0.406]` | Per-channel mean (ImageNet defaults) |
| `normalize_std` | `[0.229, 0.224, 0.225]` | Per-channel std (ImageNet defaults) |

---

## Model Variants

| Variant | Params | Frames | Resolution |
|---------|--------|--------|------------|
| `vjepa2-vitl-fpc64-256` | ~307M | 64 | 256px |
| `vjepa2-vith-fpc64-256` | ~632M | 64 | 256px |
| `vjepa2-vitg-fpc64-256` | ~1.1B | 64 | 256px |

---

## Cost Estimate (Modal.com)

| Operation | Resource | Time | Est. Cost |
|-----------|----------|------|-----------|
| Model download | A100 40GB | ~5 min | ~$0.10 |
| Preprocess (1 video) | 4 CPU | ~10s | ~$0.001 |
| Encode — Standard (A10G) | A10G | ~4s | ~$0.002 |
| Encode — Premium (A100) | A100 40GB | ~2s | ~$0.004 |
| JEDi score (10 vs 10) | A100 40GB | ~25s | ~$0.015 |
| JEDi pipeline | A100 40GB | ~20s | ~$0.010 |
| SurPrise score (1 video) | A100 40GB | ~5s | ~$0.001 |
| SurPrise pipeline | A100 40GB | ~15s | ~$0.007 |

*Approximate — based on Modal's A100 ($0.000222/s) and A10G ($0.000116/s) rates.*

---

## Requirements

- Python 3.12+
- Modal account & CLI (`pip install modal && modal setup`)
- HuggingFace account (for gated model access if required)

Python dependencies are installed automatically inside the Modal image — no local `pip install` needed for deployment.

---

## Local Preprocessing (No GPU)

```bash
pip install opencv-python numpy torch
python preprocess.py my_video.mp4 16 224
# → prints tensor shape, sampled frame indices, video metadata
```

---

## References

- [V-JEPA 2 Paper (Meta AI)](https://arxiv.org/abs/2506.09985)
- [V-JEPA 2 GitHub](https://github.com/facebookresearch/vjepa2)
- [JEDi Paper — arXiv:2410.05203](https://arxiv.org/abs/2410.05203)
- [JEDi GitHub](https://github.com/oooolga/JEDi)
- [VideoJEDi PyPI](https://pypi.org/project/VideoJEDi/)
- [SurPrise Paper — arXiv:2502.11831](https://arxiv.org/abs/2502.11831)
- [hyle-surprice-metric-evaluator](https://github.com/EroPerez/hyle_surprise_metric_evaluator)
- [HuggingFace: facebook/vjepa2](https://huggingface.co/collections/facebook/v-jepa-2)
- [Redsys TPV Virtual Documentation](https://pagosonline.redsys.es/desarrolladores-inicio/continar-integracion/#)
- [Modal Documentation](https://modal.com/docs)
