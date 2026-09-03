# Pinned to the node:20-slim digest checked on 2026-09-03.
FROM node:20-slim@sha256:2cf067cfed83d5ea958367df9f966191a942351a2df77d6f0193e162b5febfc0

WORKDIR /app

COPY package.json package-lock.json* ./
RUN npm ci

COPY . .
RUN npm run build && chown -R node:node /app

# Drop root before the server runs. node:20-slim ships this user already.
USER node

EXPOSE 3000

# node:20-slim has no curl, so the healthcheck uses Node's built-in fetch.
HEALTHCHECK --interval=15s --timeout=5s --start-period=20s --retries=5 \
  CMD node -e "fetch('http://localhost:3000').then((r) => process.exit(r.ok ? 0 : 1)).catch(() => process.exit(1))"

CMD ["npm", "start"]
