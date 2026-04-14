# Move this app to another repository + deploy on Vercel

## 1) Create a new GitHub repository
- In GitHub: **New repository**
- Name suggestion: `bharat-political-strategy-lab`
- Keep it empty (no README/license/gitignore) for easiest push

## 2) Push this codebase into that new repo
From your local clone of this project:

```bash
git remote -v
git remote add target https://github.com/<your-user>/<new-repo>.git
git push target HEAD:main
```

If your target repo already has commits, use:

```bash
git push target HEAD:main --force-with-lease
```

## 3) Verify required files are present
- `package.json`
- `app/` (Next App Router)
- `components/`
- `lib/`
- `data/`
- `next.config.mjs`
- `tailwind.config.ts`
- `tsconfig.json`

## 4) Deploy to Vercel
1. Login to https://vercel.com
2. Click **Add New → Project**
3. Select the new GitHub repository
4. Build settings:
   - **Framework preset**: Next.js
   - **Build command**: `npm run build`
   - **Install command**: `npm install`
5. No environment variables are required
6. Click **Deploy**

## 5) Check post-deploy endpoints
- `/api/health`
- `/api/catalog?page=1&pageSize=12`
- `/api/scenarios`

## 6) Free-tier storage strategy
- Primary: local JSON files in `data/` (zero-cost)
- Optional backup: commit periodic snapshots to GitHub
- Optional object backup: Cloudflare R2 free tier

## 7) If deploy fails due package install restrictions
- Confirm npm registry access in your network
- Re-run deployment from Vercel cloud (normally has registry access)
- Ensure lockfile consistency if adding one later (`package-lock.json`)
