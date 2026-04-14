# Bharat Political Strategy Lab

Production-ready MVP for a local, free-tier political strategy simulation platform focused on India.

## Features implemented
- Strategic foresight engine (best/probable/worst + confidence)
- Adversarial simulator (Party A, Party B, media, public rounds)
- Policy stress testing with thresholds/flags/scenarios
- Narrative war simulator (pro/opposition/media)
- Time-shift simulation with approval curve
- Political DNA segment mapping and weighted contribution
- Reality vs perception gap analyzer
- Manifesto auto-optimizer (promise extraction proxy, contradiction checks)
- Simulation memory and scenario comparison
- Unified workflow in a multi-panel dashboard
- Neural Network Layer (local in-process MLP for score/risk/stability deltas)
- Self-correcting neural memory (`data/neural_memory.json`) for incremental learning
- AI explanation bot layer for plain-language interpretation and strategic Q&A
- Monte Carlo uncertainty layer (P10/P50/P90 + win probability)
- Message Testing Lab (persuasion vs backlash for framing variants)
- Supporter funnel analytics layer (awareness→consideration→support→turnout)
- Consensus map layer with bridging narrative for cross-bloc coalition messaging
- Campaign commitment tracker (issue-ticket layer for follow-up accountability)

## Data and opinion layer (2000–2025)
- Local free datasets in `/data`
- Yearly sentiment index, topic-wise sensitivity, regional proxy
- Designed for manual CSV augmentation from ECI/Census/data.gov.in/Google Trends exports

## Tech stack
- Next.js App Router
- Tailwind CSS
- Framer Motion
- Recharts
- Zustand
- Local JSON persistence (`data/simulations.json`)

## Run locally
```bash
npm install
npm run dev
```

Open http://localhost:3000

## Build for production
```bash
npm run build
npm run start
```

## Free deployment to Vercel (no env vars)
1. Create a new GitHub repository and push this code.
2. Sign in to [Vercel](https://vercel.com) with GitHub.
3. Click **Add New → Project**.
4. Import your repository.
5. Build settings:
   - Framework Preset: **Next.js**
6. Click **Deploy**.

No environment variables are required.

## Recommended versions (April 2026)
- Node.js 24 LTS
- Next.js 16 LTS (latest patch)
- React 19.x

See `VERSION_GUIDE.md` for details.

## Free storage options
- Default: local JSON files in `data/` (zero-cost, simplest).
- Optional free cloud sync: push exported JSON snapshots to GitHub repo storage.
- Optional: Cloudflare R2 free tier for low-volume object backups.

## New platform APIs
- `GET /api/health` for dataset and free-tier diagnostics.
- `GET /api/catalog?page=1&pageSize=12&q=` for 4K+ scenario browsing and search.


## Offline resilience mode
- App now uses local snapshot memory for runs/scenarios when network/API is unavailable.
- If simulation API is unreachable, a synthetic offline-safe run is generated to keep UX unblocked.
- Offline banner clearly indicates reduced-fidelity mode until network is restored.


## API reference for product users
- Human-readable API guide: `docs/API_REFERENCE.md`
- Machine-readable endpoint index: `GET /api/docs`


