ARG NODE_BASE_IMAGE=node:20-alpine
FROM ${NODE_BASE_IMAGE} AS build

ARG VITE_API_KEY=
ARG VITE_API_BASE_URL=
ARG VITE_DEFAULT_LLM_API_URL=
ARG VITE_LLM_API_URLS=
ARG VITE_SUPABASE_URL=
ARG VITE_SUPABASE_ANON_KEY=

WORKDIR /app

COPY frontend-workflow/package.json frontend-workflow/package-lock.json ./
RUN npm ci

# 复制前端源码；若存在 frontend-workflow/.env，Vite build 也会读取。
COPY frontend-workflow/ ./

RUN cp -f .env.example .env.production || true && \
    rm -f .env.production.local && \
    touch .env.production.local && \
    if [ -n "$VITE_API_KEY" ]; then echo "VITE_API_KEY=$VITE_API_KEY" >> .env.production.local; fi && \
    if [ -n "$VITE_API_BASE_URL" ]; then echo "VITE_API_BASE_URL=$VITE_API_BASE_URL" >> .env.production.local; fi && \
    if [ -n "$VITE_DEFAULT_LLM_API_URL" ]; then echo "VITE_DEFAULT_LLM_API_URL=$VITE_DEFAULT_LLM_API_URL" >> .env.production.local; fi && \
    if [ -n "$VITE_LLM_API_URLS" ]; then echo "VITE_LLM_API_URLS=$VITE_LLM_API_URLS" >> .env.production.local; fi && \
    if [ -n "$VITE_SUPABASE_URL" ]; then echo "VITE_SUPABASE_URL=$VITE_SUPABASE_URL" >> .env.production.local; fi && \
    if [ -n "$VITE_SUPABASE_ANON_KEY" ]; then echo "VITE_SUPABASE_ANON_KEY=$VITE_SUPABASE_ANON_KEY" >> .env.production.local; fi

RUN npm run build

ARG NGINX_BASE_IMAGE=nginx:alpine
FROM ${NGINX_BASE_IMAGE}

COPY frontend-workflow/nginx.conf.template /etc/nginx/templates/default.conf.template
COPY frontend-workflow/docker-nginx-start.sh /docker-nginx-start.sh
RUN chmod +x /docker-nginx-start.sh
COPY --from=build /app/dist /usr/share/nginx/html

EXPOSE 80
CMD ["/docker-nginx-start.sh"]
