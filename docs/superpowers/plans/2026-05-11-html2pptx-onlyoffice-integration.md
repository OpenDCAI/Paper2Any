# html2pptx + OnlyOffice Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a new Paper2Any HTML-based editable PPTX export path that keeps the existing HTML generation flow, converts the generated HTML through `html2pptx/package`, uploads the PPTX to the backend outputs store, and opens that PPTX in ONLYOFFICE for online editing.

**Architecture:** Paper2Any will keep producing its existing frontend HTML slide deck. We will add a new HTML deck artifact that is saved and listed as an output, then add a browser-side `dom-to-pptx` export helper that turns that HTML into a real PPTX blob. The blob will be uploaded through the existing `/api/v1/files/upload` path so the backend can serve it under `/outputs/...`, and new backend ONLYOFFICE endpoints on `files.py` will generate config, stream the document, and persist callback saves.

**Tech Stack:** FastAPI, existing Paper2Any React/Vite frontend, TypeScript, Python, ONLYOFFICE Document Server, `html2pptx/package` (`dom-to-pptx` browser bundle), existing backend file upload and output URL helpers.

---

### Task 1: Add an HTML deck artifact and browser-side html2pptx export helper

**Files:**
- Create: `frontend-workflow/src/components/paper2ppt/htmlDeckArtifact.ts`
- Create: `frontend-workflow/src/components/paper2ppt/html2pptxExport.ts`
- Modify: `frontend-workflow/src/components/paper2ppt/FrontendCompleteStep.tsx`
- Modify: `frontend-workflow/src/components/paper2ppt/index.tsx`
- Create: `frontend-workflow/src/components/paper2ppt/__tests__/htmlDeckArtifact.test.ts`

- [ ] **Step 1: Write the failing test**

```ts
import { describe, expect, it } from 'vitest';
import { buildHtmlDeckArtifact } from '../htmlDeckArtifact';

describe('buildHtmlDeckArtifact', () => {
  it('keeps the slide root and writes a complete html document', () => {
    const html = buildHtmlDeckArtifact([
      '<section class="slide-root"><h1>Slide 1</h1></section>',
      '<section class="slide-root"><h1>Slide 2</h1></section>',
    ]);

    expect(html).toContain('<!doctype html>');
    expect(html).toContain('slide-root');
    expect(html).toContain('Slide 1');
    expect(html).toContain('Slide 2');
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd frontend-workflow && npx vitest run src/components/paper2ppt/__tests__/htmlDeckArtifact.test.ts`
Expected: FAIL because `buildHtmlDeckArtifact` does not exist yet.

- [ ] **Step 3: Write minimal implementation**

```ts
export function buildHtmlDeckArtifact(slideHtmlBlocks: string[]): string {
  return `<!doctype html><html><head><meta charset="utf-8"></head><body>${slideHtmlBlocks.join('')}</body></html>`;
}
```

- [ ] **Step 4: Add the browser export helper**

Implement `html2pptxExport.ts` so the browser can:
1. Load the bundled `dom-to-pptx` script from a static asset path
2. Render the generated HTML deck into a hidden container
3. Call `exportToPptx([...slideRoots], { skipDownload: true, svgAsVector: true })`
4. Return a `Blob` for upload

Use the existing slide HTML renderer in `frontend-workflow/src/components/paper2ppt/frontendSlideUtils.ts` instead of inventing a second HTML dialect.

- [ ] **Step 5: Wire the new button into the frontend complete step**

Add a second completion action next to the existing `生成可编辑 PPTX` flow:
- `导出 HTML 可编辑 PPTX`
- `在线编辑 PPTX`

The export flow should:
1. Generate the HTML deck artifact blob
2. Upload the HTML artifact through `uploadGeneratedResultBlob`
3. Run the html2pptx conversion
4. Upload the resulting PPTX blob through `uploadGeneratedResultBlob`
5. Save the uploaded PPTX URL for the ONLYOFFICE editor button

- [ ] **Step 6: Run the frontend test again**

Run: `cd frontend-workflow && npx vitest run src/components/paper2ppt/__tests__/htmlDeckArtifact.test.ts`
Expected: PASS.

### Task 2: Add backend ONLYOFFICE support for uploaded PPTX files

**Files:**
- Modify: `fastapi_app/config/settings.py`
- Create: `fastapi_app/services/onlyoffice_file_service.py`
- Modify: `fastapi_app/routers/files.py`
- Create: `tests/test_files_onlyoffice.py`

- [ ] **Step 1: Write the failing backend test**

```python
from pathlib import Path

def test_onlyoffice_config_and_callback_round_trip(tmp_path, monkeypatch):
    ...
```

The test should assert:
1. A PPTX path under `outputs/...` yields an ONLYOFFICE config payload
2. The config includes a document download URL and callback URL
3. The callback writes the saved PPTX back to the same path

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest -q tests/test_files_onlyoffice.py`
Expected: FAIL because the new service and routes do not exist yet.

- [ ] **Step 3: Implement the ONLYOFFICE file service**

`onlyoffice_file_service.py` should:
1. Resolve and validate an authenticated user-owned output path
2. Build a stable document key from file path, mtime, size, and optional editor session id
3. Return the Document Server script URL, config payload, and download/callback URLs
4. Accept callback saves and replace the stored PPTX atomically

Reuse the existing output path helpers and the existing file access token utilities in `files.py` rather than adding a second path-security system.

- [ ] **Step 4: Expose the ONLYOFFICE routes on `files.py`**

Add routes under `/api/v1/files`:
- `GET /onlyoffice/config`
- `GET|HEAD /onlyoffice/download/{document_key}.pptx`
- `POST /onlyoffice/callback`

These routes should work with the uploaded PPTX file from the html2pptx export step.

- [ ] **Step 5: Add config defaults**

Add settings fields and environment examples for:
- `ONLYOFFICE_DOCUMENT_SERVER_URL`
- `ONLYOFFICE_THINKFLOW_PUBLIC_URL`
- `ONLYOFFICE_DOCUMENT_DOWNLOAD_BASE_URL`
- `ONLYOFFICE_JWT_SECRET`

- [ ] **Step 6: Run the backend test again**

Run: `pytest -q tests/test_files_onlyoffice.py`
Expected: PASS.

### Task 3: Hook Paper2Any frontend results into the new export and editor flow

**Files:**
- Modify: `frontend-workflow/src/components/paper2ppt/index.tsx`
- Modify: `frontend-workflow/src/components/paper2ppt/FrontendCompleteStep.tsx`
- Modify: `frontend-workflow/src/services/fileService.ts`
- Modify: `frontend-workflow/vite.config.ts`

- [ ] **Step 1: Add the new state and handlers**

Track:
- the saved HTML artifact URL
- the uploaded html2pptx PPTX URL
- the ONLYOFFICE session id
- loading/error state for export and editor launch

The editor launch handler should call the new backend ONLYOFFICE config route with the uploaded PPTX path and open an embedded modal/iframe using the returned `script_url` and `config`.

- [ ] **Step 2: Make the HTML artifact visible in the result list**

After the frontend HTML deck is generated, upload the HTML artifact so it shows up alongside the other `/outputs/...` files. That keeps the HTML deck as a first-class result item and gives the user something concrete to convert.

- [ ] **Step 3: Proxy ONLYOFFICE through Vite**

Add a `/onlyoffice` proxy in `frontend-workflow/vite.config.ts` that points at the local Document Server container, matching the ThinkFlow deployment pattern.

- [ ] **Step 4: Add a frontend behavior test or smoke check**

If there is already a frontend test harness for this area, add a small test that the new export/editor buttons render in frontend mode when an exported PPTX exists. If the repo does not have a practical browser test for this component, keep the coverage focused on the helper tests from Task 1 and the backend tests from Task 2, then verify the UI manually during the smoke run.

### Task 4: Verify the end-to-end flow with a real HTML deck

**Files:**
- No new files; validate the integrated runtime behavior

- [ ] **Step 1: Run the relevant backend tests**

Run:
`pytest -q tests/test_paper2ppt_generate_route.py tests/test_files_onlyoffice.py`

Expected: all pass.

- [ ] **Step 2: Build the frontend**

Run:
`cd frontend-workflow && npm run build`

Expected: clean build with the new html2pptx helper and ONLYOFFICE proxy config.

- [ ] **Step 3: Generate one frontend Paper2PPT deck and export it**

Run the Paper2PPT frontend workflow, generate a real HTML deck, export the new editable PPTX path, and confirm the uploaded PPTX appears under the user outputs list.

- [ ] **Step 4: Open the ONLYOFFICE editor**

Confirm the new `在线编辑 PPTX` action loads the editor, the document opens from `/outputs/...`, and saving writes back to the same PPTX file.

