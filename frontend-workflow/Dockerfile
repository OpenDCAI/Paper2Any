FROM node:20-alpine AS build

WORKDIR /app

COPY frontend-workflow/package.json frontend-workflow/package-lock.json ./
RUN npm ci

# 复制前端源码（包含 .env 文件，Vite build 时会自动读取）
COPY frontend-workflow/ ./

RUN npm run build

FROM nginx:alpine

COPY frontend-workflow/nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=build /app/dist /usr/share/nginx/html

EXPOSE 80
