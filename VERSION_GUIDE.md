# Version Guide (April 2026)

For new production deployments, use **Next.js 16 LTS** (latest patch), not 14.x.

## Recommended baseline
- **Node.js:** 24 LTS
- **Next.js:** 16.2.3 (or latest 16.x patch)
- **React / React DOM:** 19.x (matching Next.js 16 peer requirements)
- **TypeScript:** 5.6+

## Why this baseline
- Next.js 16 is the active LTS line and receives security + bug-fix patches.
- Next.js 15 remains supported only until **October 21, 2026**.
- Next.js 14 support already ended.

## Upgrade policy for this repo
- Keep major versions stable for a sprint (e.g., 16.x), but always upgrade to the latest patch release.
- Re-run production checks after every framework minor upgrade.

## Quick check commands
```bash
node -v
npm outdated
npx next --version
```
