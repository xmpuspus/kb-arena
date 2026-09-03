# Pinned to the node:20-slim digest checked on 2026-09-03.
FROM node:20-slim@sha256:2cf067cfed83d5ea958367df9f966191a942351a2df77d6f0193e162b5febfc0

WORKDIR /app

COPY package.json package-lock.json* ./
RUN npm ci

COPY . .

# next.config.mjs sets output: "export", so the static bundle below has no
# server to read env vars from at runtime. Next.js inlines NEXT_PUBLIC_*
# vars into the JS at build time, so the backend URL has to arrive as a
# build ARG, not a compose environment: entry (that runs too late).
ARG NEXT_PUBLIC_API_URL=http://localhost:8000
ENV NEXT_PUBLIC_API_URL=$NEXT_PUBLIC_API_URL

RUN npm run build && chown -R node:node /app

# "next start" cannot run against a static export, and fails on every boot
# (Next.js's own error names serve as the fix), so a static file server
# ships the export instead. Pinned, so a new serve release cannot change
# what a rebuild of this exact commit runs.
RUN npm install -g serve@14.2.6

# Drop root before the server runs. node:20-slim ships this user already.
USER node

EXPOSE 3000

# node:20-slim has no curl, so the healthcheck uses Node's built-in fetch.
HEALTHCHECK --interval=15s --timeout=5s --start-period=20s --retries=5 \
  CMD node -e "fetch('http://localhost:3000').then((r) => process.exit(r.ok ? 0 : 1)).catch(() => process.exit(1))"

CMD ["serve", "-s", "out", "-l", "3000"]
