# API Reference — Bharat Political Strategy Lab

This software product is designed for both direct UI use and API-first integration.

## Base URL
- Local: `http://localhost:3000`
- Production: `https://<your-vercel-domain>`

---

## 1) `GET /api/health`
Platform diagnostics and free-tier flags.

### Example
```bash
curl -s http://localhost:3000/api/health | jq
```

---

## 2) `GET /api/catalog?page=1&pageSize=12&q=`
Searchable, paginated scenario catalog.

### Query Params
- `page` (default `1`)
- `pageSize` (default `12`, range `5-50`)
- `q` optional search string

### Example
```bash
curl -s "http://localhost:3000/api/catalog?page=1&pageSize=5&q=jobs" | jq
```

---

## 3) `GET /api/history`
Historical sentiment (2000–2025) + topic and regional proxies.

### Example
```bash
curl -s http://localhost:3000/api/history | jq
```

---

## 4) `GET /api/scenarios`
Reads persisted simulation runs.

### Example
```bash
curl -s http://localhost:3000/api/scenarios | jq
```

---

## 5) `POST /api/simulate`
Runs the complete simulation stack and persists result.

### Request JSON
```json
{
  "title": "Fuel tax rationalization",
  "policyText": "Rationalize fuel taxes and redirect budget to transport subsidy",
  "launchWindow": "6m",
  "regionFocus": "national",
  "visibility": 72,
  "estimatedCost": 18000,
  "beneficiariesM": 21,
  "manifestoText": "optional"
}
```

### Example
```bash
curl -s -X POST http://localhost:3000/api/simulate \
  -H "Content-Type: application/json" \
  -d '{"title":"Test","policyText":"Jobs support package","launchWindow":"3m","regionFocus":"north","visibility":68,"estimatedCost":25000,"beneficiariesM":18}' | jq
```

---

## 6) `POST /api/manifesto`
Manifesto-only analysis helper.

### Example
```bash
curl -s -X POST http://localhost:3000/api/manifesto \
  -H "Content-Type: application/json" \
  -d '{"manifestoText":"Lower taxes; increase spending on welfare and jobs"}' | jq
```

---

## 7) `GET /api/docs`
Machine-readable API directory (JSON) for direct product consumers.

### Example
```bash
curl -s http://localhost:3000/api/docs | jq
```

---

## Notes
- No paid APIs required.
- Works with local JSON persistence.
- Offline fallback mode exists for UI continuity (synthetic fallback run if API unavailable).
