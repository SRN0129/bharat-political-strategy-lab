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
