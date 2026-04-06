Deployment profiles are split into:

- `*.env.example`: tracked examples for release
- `*.env`: local machine overrides, ignored by git

Ownership boundary:

- `fastapi_app/.env`: backend business config and third-party credentials
- `frontend-workflow/.env`: frontend-visible defaults (`VITE_*`)
- `deploy/profiles/*.env`: machine/deploy-only settings such as python path, ports, GPU layout, local model paths

Do not duplicate the same API URL / API key across deploy profiles and app `.env` files.
Deploy profiles should not carry frontend default models or third-party business API credentials.

Typical setup:

```bash
cp deploy/profiles/muxi.env.example deploy/profiles/muxi.env
cp deploy/profiles/nv.env.example deploy/profiles/nv.env
```

Only fill the values that your machine actually needs. Empty values fall back to
auto-discovery in the start scripts.
