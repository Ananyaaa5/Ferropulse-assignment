# Ferropulse Internship Assignment

**Submitted by:** Ananya V Akkihal
**Role Applied:** AI Intern / Backend & Mobile Development Intern

---

## Questions Selected

| Category | Question | File |
|----------|----------|------|
| Group 1 — Technical / Coding / SQL | Q5: Round Robin Delivery Allocation | `g1_q5_rr.py` |
| Group 2 — System Design / Analytics / AI | Q5: AI Recommendation with Prompt Optimization + Semantic Caching | `g2_q5_ai_recommendation.py` |

---

## How to Run

```bash
# 1. Clone the repository
git clone https://github.com/Ananyaaa5/Ferropulse-assignment.git
cd Ferropulse-assignment

# 2. Create and activate a virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Mac/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run Group 1
python g1_q5_rr.py

# 5. Run Group 2
#    Optionally set a Gemini API key for live LLM responses
#    (demo runs with realistic mock responses if no key is set)
#    PowerShell: $env:GEMINI_API_KEY = "your_key_here"
python g2_q5_ai_recommendation.py
```

---

## Group 1 — Q5: Round Robin Delivery Allocation

### Problem Summary
Assign incoming delivery orders to active delivery partners fairly using round-robin scheduling. Partners can go online or offline at any time, and orders must only go to currently active partners.

### Approach

A single **ordered list + integer pointer** drives the round-robin cycle:

```
Active Queue:  [Ravi, Suresh, Meena]
Pointer:            ↑ (next to assign)
```

- `go_online()` — appends the partner to the active queue.
- `go_offline()` — removes the partner and **adjusts the pointer** so the cycle doesn't skip or repeat a position.
- `assign_order()` — picks `queue[pointer]`, then increments and wraps the pointer.

### Complexity

| Operation | Time | Space |
|-----------|------|-------|
| `go_online` | O(1) | — |
| `assign_order` | O(1) | — |
| `go_offline` | O(n) — list removal | O(p) total for p partners |

### Sample Output
```
🟢 Ravi came online    [Active partners: 1]
🟢 Suresh came online  [Active partners: 2]
🟢 Meena came online   [Active partners: 3]

📦 ORD-001 → Ravi
📦 ORD-002 → Suresh
📦 ORD-003 → Meena
📦 ORD-004 → Ravi
📦 ORD-005 → Suresh
📦 ORD-006 → Meena

🔴 Suresh went offline  [Active partners: 2]

📦 ORD-007 → Meena      ← cycle resumes correctly without skipping
📦 ORD-008 → Ravi
📦 ORD-009 → Meena
📦 ORD-010 → Ravi

🟢 Arjun came online   [Active partners: 3]

📦 ORD-011 → Meena
📦 ORD-012 → Arjun
📦 ORD-013 → Ravi

⚠️  ORD-014 could not be assigned — no active partners!

═════════════════════════════════════════════
  DELIVERY ALLOCATION SUMMARY
═════════════════════════════════════════════
  Partner              Orders   Status
  ─────────────────────────────────────────
  Ravi                      5   🔴 Offline
  Suresh                    2   🔴 Offline
  Meena                     5   🔴 Offline
  Arjun                     1   🔴 Offline
═════════════════════════════════════════════
  Total orders assigned: 13
  Max load imbalance:    4 order(s)
═════════════════════════════════════════════
```

---

## Group 2 — Q5: AI Recommendation System with Prompt Optimization + Semantic Caching

### Problem Summary
A recommendation system that sends user data to an LLM must solve two problems:
1. **High token usage** from sending full raw user profiles (~2000–5000 tokens per request).
2. **Repeated API calls** for the same or similar queries from different users, wasting cost and time.

### Architecture

```
Incoming Request (user_data + query)
         │
         ▼
┌─────────────────────┐
│  UserDataSummarizer │  Compresses raw profile → ~150 tokens
│  (Token Optimizer)  │  Extracts: top items, cuisines,
└──────────┬──────────┘  avg spend, price sensitivity, recency
           │
           ▼
┌─────────────────────┐
│    SemanticCache    │  Layer 1: Exact hash match (O(1))
│    (Lookup)         │  Layer 2: Fuzzy similarity match
└──────┬──────┬───────┘           (SequenceMatcher ratio ≥ 0.82)
    HIT│      │MISS
       │      ▼
       │  ┌───────────────────────┐
       │  │  Build Prompt         │  Only compressed summary in prompt
       │  │  → Call LLM API       │  (Gemini / OpenAI / local LLM)
       │  │  → Store in Cache     │
       │  └───────────────────────┘
       ▼
   Return Response  { source: "cache" | "llm" }
```

### Token Optimization Strategy
Raw user data (full order history, search logs, behaviour events) can run 2000–5000 tokens per request. The `UserDataSummarizer` reduces this to ~100–200 tokens by:
- Taking only the **top-N most ordered items/restaurants** via frequency analysis
- Including only the **last 5 orders** for recency signal
- Deriving a single **price sensitivity label** (budget / mid-range / premium) from average spend
- Keeping the **last 3 search terms** for intent signal

This preserves ~90% of the recommendation-relevant signal at ~5–10% of the token cost.

### Semantic Caching Strategy

**User Segmentation** — Cache is bucketed by `price_sensitivity:top_cuisines`. This prevents a budget South-Indian user from receiving recommendations cached for a premium Chinese-food user.

**Layer 1 — Exact Match**: Queries are normalized (lowercased, words sorted alphabetically) before hashing with MD5. This means `"spicy dinner suggestions"` and `"dinner spicy suggestions"` resolve to the same cache key.

**Layer 2 — Fuzzy Match**: Uses Python's `difflib.SequenceMatcher` ratio. A threshold of **0.82** catches clear paraphrases (`"suggest spicy food"` ≈ `"recommend something spicy"`) while rejecting structurally different queries.

**Eviction**: LFU (Least Frequently Used) — removes the entry with the fewest hits when cache capacity is reached.

**Fallback Design**: The system gracefully falls back to a realistic mock response if the LLM API is unavailable (quota exceeded, no key set, network error). This ensures the demo always runs end-to-end and reflects production-grade resilience.

### LLM Integration
The system is built to be LLM-agnostic. To plug in a real model, set the `GEMINI_API_KEY` environment variable and the `_call_llm` method will automatically use Gemini. The same method can be swapped for any other provider (OpenAI, Anthropic, local Ollama) with minimal changes.

### Sample Output
```
═══════════════════════════════════════════════════════
  AI RECOMMENDATION SYSTEM — DEMO
═══════════════════════════════════════════════════════

  Request: "suggest something spicy for dinner"  (user: U001)
  [Summarizer] Compressed user profile | ~105 tokens saved
  [Cache] ❌ Miss
  [LLM]  🤖 Calling Gemini API...
  [Cache] 💾 Stored response
  → Source: llm | Confidence: 0.91
  → Top pick: Nagarjuna – Meals + Chilly Chicken

  Request: "suggest something spicy for dinner"  (user: U002)
  [Summarizer] Compressed user profile | ~60 tokens saved
  [Cache] ✅ Exact hit    ← No LLM call!
  → Source: cache

  Request: "recommend spicy dinner options"  (user: U001)
  [Cache] ✅ Fuzzy hit (score=0.84)    ← No LLM call!
  → Source: cache

═══════════════════════════════════════════════════════
  SYSTEM PERFORMANCE STATS
═══════════════════════════════════════════════════════
  Total requests  : 5
  Actual LLM calls: 2
  Cache hits      : 3
  Cache hit rate  : 60.0%
  Cached entries  : 2
═══════════════════════════════════════════════════════
```

---

## Project Structure

```
Ferropulse-assignment/
├── g1_q5_rr.py   # Round Robin Delivery Allocator
├── g2_q5_ai_recommendation.py     # AI Recommendation System
├── requirements.txt                   # Python dependencies
├── .gitignore                         # Excludes venv, cache, secrets
└── README.md                          # This file
```

---

## Language & Dependencies

- **Language**: Python 3.10+
- **Standard library**: `hashlib`, `difflib`, `collections`, `dataclasses`, `os` — no install needed for Group 1 or the Group 2 demo
- **Optional** (for live LLM in Group 2): `google-genai`

```bash
pip install -r requirements.txt
```
