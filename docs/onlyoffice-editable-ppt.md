# Paper2Any ONLYOFFICE Editable PPTX

Paper2Any 的 html2pptx 可编辑导出可以接入 ONLYOFFICE Document Server，生成 PPTX 后直接在线编辑。未配置 ONLYOFFICE 时，前端仍可下载可编辑 PPTX。

## html2pptx Converter

HTML 到可编辑 PPTX 的转换使用 vendored `dom-to-pptx` 浏览器 bundle：

- Bundle：`frontend-workflow/public/vendor/dom-to-pptx.bundle.js`
- License：`frontend-workflow/public/vendor/dom-to-pptx.LICENSE`
- 上游：`https://github.com/atharva9167j/dom-to-pptx`
- 许可证：MIT

当前采用静态 bundle 是为了在前端稳定加载 `html2pptx` 转换器，并避免开发/部署环境单独发布本地 package。后续升级该库时，应同步更新 bundle 和 license 文件。

## Required Settings

在 `fastapi_app/.env` 或部署环境中配置：

```bash
ONLYOFFICE_DOCUMENT_SERVER_URL=/onlyoffice
ONLYOFFICE_THINKFLOW_PUBLIC_URL=http://host.docker.internal:8000
ONLYOFFICE_DOCUMENT_DOWNLOAD_BASE_URL=http://host.docker.internal:8000
ONLYOFFICE_SERVER_DOWNLOAD_URL_BASE=http://127.0.0.1:8082
ONLYOFFICE_JWT_SECRET=
```

- `ONLYOFFICE_DOCUMENT_SERVER_URL`：浏览器加载 ONLYOFFICE 的入口。本地 Vite 开发建议使用 `/onlyoffice`，由 `frontend-workflow/vite.config.ts` 代理到 `http://localhost:8082`。
- `ONLYOFFICE_THINKFLOW_PUBLIC_URL`：ONLYOFFICE 容器可访问的 Paper2Any 后端地址，用于保存回调。
- `ONLYOFFICE_DOCUMENT_DOWNLOAD_BASE_URL`：ONLYOFFICE 容器回源下载 PPTX 的后端地址。本地 Docker 场景推荐 `http://host.docker.internal:8000`，不要配成浏览器里的 `localhost:3000`。
- `ONLYOFFICE_SERVER_DOWNLOAD_URL_BASE`：Paper2Any 后端下载 ONLYOFFICE 保存结果时使用的 Document Server 地址。本地 Vite/SSH 转发场景推荐 `http://127.0.0.1:8082`，用于把回调 payload 里的 `http://localhost:13000/onlyoffice/cache/...` 重写为后端可访问的 `http://127.0.0.1:8082/cache/...`。
- `ONLYOFFICE_JWT_SECRET`：仅在 Document Server 开启 JWT 时填写，并保持与 Document Server 一致。本地调试默认留空，同时容器使用 `JWT_ENABLED=false`。

URL 角色需要分清：

- `document.url` 和 `callbackUrl` 是 Document Server 容器访问后端用的 URL，应该能从容器内访问。
- `storage.externalHost` 是 Document Server 返回给浏览器的缓存资源 URL，应该是浏览器当前可访问的前端代理地址，例如 `http://localhost:3000/onlyoffice`。
- callback payload 里的保存结果 `url` 可能跟随 `storage.externalHost`，因此 SSH 转发时会是浏览器本机地址。后端保存前会用 `ONLYOFFICE_SERVER_DOWNLOAD_URL_BASE` 改写这个 URL。

## Local Docker Deployment

如果本机没有 `onlyoffice/documentserver:latest` 镜像，可以从已准备好的 tar 包导入：

```bash
docker load -i /mnt/paper2any/dingcheng/onlyoffice-documentserver-latest.tar
```

启动 Document Server：

```bash
docker run -d --name paper2any-onlyoffice \
  -p 8082:80 \
  --add-host=host.docker.internal:host-gateway \
  -e JWT_ENABLED=false \
  -e ALLOW_PRIVATE_IP_ADDRESS=true \
  onlyoffice/documentserver:latest
```

如果容器已存在，先停止并移除旧容器后再启动：

```bash
docker stop paper2any-onlyoffice
docker rm paper2any-onlyoffice
```

Paper2Any 前端本地端口默认是 `3000`，`frontend-workflow/vite.config.ts` 已将 `/onlyoffice` 代理到 `http://localhost:8082`。为了让 ONLYOFFICE 编辑器内部的缓存资源也走前端同源代理，需要修改容器内 `local.json`：

```bash
docker cp paper2any-onlyoffice:/etc/onlyoffice/documentserver/local.json /tmp/paper2any-onlyoffice-local.json
python - <<'PY'
import json
from pathlib import Path

path = Path("/tmp/paper2any-onlyoffice-local.json")
data = json.loads(path.read_text())
storage = data.setdefault("storage", {})
storage["externalHost"] = "http://localhost:3000/onlyoffice"
storage["useDirectStorageUrls"] = False
path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
PY
docker cp /tmp/paper2any-onlyoffice-local.json paper2any-onlyoffice:/etc/onlyoffice/documentserver/local.json
docker exec paper2any-onlyoffice supervisorctl restart ds:docservice ds:converter
```

如果前端是通过 SSH 端口转发打开的，`externalHost` 要写浏览器实际访问的地址。例如本机访问 `http://localhost:13000`、远端 Vite 仍是 `3000` 时，应使用：

```bash
storage["externalHost"] = "http://localhost:13000/onlyoffice"
```

不要把 `local.json` 以只读 bind mount 的方式挂进容器。Document Server 启动脚本会写这个文件；只读挂载会导致 `EBUSY`/`Read-only file system`，进而让 JWT 或缓存 URL 配置处在不可信状态。推荐按上面的流程：容器启动完成后 `docker cp` 配置进去，再重启 `ds:docservice ds:converter`。

## Paper2Any Endpoints

在线编辑使用这些后端接口：

- `GET /api/v1/files/onlyoffice/config`
- `GET|HEAD /api/v1/files/onlyoffice/download/{document_key}.pptx`
- `POST /api/v1/files/onlyoffice/callback`

编辑完成后，Document Server 回调会把保存后的 PPTX 写回原输出文件路径。

## Troubleshooting

- 编辑器提示 `ONLYOFFICE_DOCUMENT_SERVER_URL is not configured`：检查后端 `.env` 是否配置并重启后端。
- ONLYOFFICE 错误码 `-4` 或下载失败：检查 `ONLYOFFICE_DOCUMENT_DOWNLOAD_BASE_URL` 是否为容器可访问的后端地址，检查 `ALLOW_PRIVATE_IP_ADDRESS=true` 是否生效。
- Ctrl+S 提示保存失败：检查后端日志里的 callback 是否返回 `{"error":1}`。如果 payload 的 `url` 是 `http://localhost:13000/onlyoffice/cache/...`，需要配置 `ONLYOFFICE_SERVER_DOWNLOAD_URL_BASE=http://127.0.0.1:8082` 并重启后端。
- 浏览器控制台报 `/onlyoffice/.../Editor.bin` 或 service worker fetch 失败：检查 `storage.externalHost` 是否为 `http://localhost:3000/onlyoffice`，并重启 `docservice` 和 `converter`。
- 如果刚重建过 Document Server，旧页面里的 `/onlyoffice/cache/...Editor.bin?...md5=...` 可能因 `storage.fs.secretString` 改变而返回 `403`。刷新页面后重新点击“在线编辑 PPTX”；前端 iframe 会清理 ONLYOFFICE 的同源 service worker/cache，避免复用旧缓存。
- JWT 报错：本地开发关闭 JWT；生产开启时，容器 JWT secret 和 `ONLYOFFICE_JWT_SECRET` 必须一致。

## Production Notes

- 生产建议将 Document Server 放在同域 HTTPS 反向代理后，并开启 JWT。
- 确保 Document Server 能访问 Paper2Any 后端的下载和回调接口。
- 不要提交本地 JWT secret、容器导出的 `local.json`、运行日志或临时 `.onlyoffice.tmp` 文件。
