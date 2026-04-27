from __future__ import annotations

import json
import os
from urllib.parse import urlparse
import re

from playwright.sync_api import Page, Route, sync_playwright


BASE_URL = os.getenv("PLAYWRIGHT_BASE_URL", "http://127.0.0.1:3101")
PNG_DATA_URL = (
    "data:image/png;base64,"
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO5w3iQAAAAASUVORK5CYII="
)


def _install_mock_api(page: Page, captured: dict[str, object]) -> None:
    runtime_config = {
        "billing_mode": "free",
        "user_api_config_required": False,
        "model_selection_locked": False,
        "managed_api_enabled": True,
        "managed_api_url": "https://api.apiyi.com/v1",
        "server_side_billing_enforced": True,
        "workflow_costs": {"image_playground": 2},
        "guest_daily_limit": 0,
        "signup_bonus_points": 0,
        "daily_grant_points": 5,
        "daily_grant_balance_cap": 15,
        "referral_inviter_points": 5,
        "referral_invitee_points": 0,
        "points_purchase_url": "",
        "points_redeem_enabled": False,
    }

    captured["quota_calls"] = 0
    captured["generate_payloads"] = []
    captured["generate_headers"] = []

    def handler(route: Route) -> None:
        path = urlparse(route.request.url).path

        if path.endswith("/api/v1/account/runtime-config"):
            route.fulfill(
                status=200,
                content_type="application/json",
                body=json.dumps(runtime_config),
            )
            return

        if path.endswith("/api/v1/account/quota"):
            captured["quota_calls"] = int(captured["quota_calls"]) + 1
            route.fulfill(
                status=200,
                content_type="application/json",
                body=json.dumps(
                    {
                        "used": 2,
                        "limit": 100,
                        "remaining": 98,
                        "is_authenticated": True,
                        "billing_mode": "free",
                    }
                ),
            )
            return

        if path.endswith("/api/v1/image-playground/generate"):
            payload = json.loads(route.request.post_data or "{}")
            captured["generate_payloads"].append(payload)
            captured["generate_headers"].append(dict(route.request.headers))
            batch_count = int(payload.get("batch_count") or 1)
            images = [
                {
                    "index": index + 1,
                    "image_url": PNG_DATA_URL,
                    "preview_url": PNG_DATA_URL,
                    "file_path": f"/tmp/generated_{index + 1}.png",
                    "preview_path": f"/tmp/generated_{index + 1}_preview.jpg",
                    "file_name": f"generated_{index + 1}.png",
                    "preview_file_name": f"generated_{index + 1}_preview.jpg",
                    "variant_label": f"Variant {index + 1}",
                }
                for index in range(batch_count)
            ]
            route.fulfill(
                status=200,
                content_type="application/json",
                body=json.dumps(
                    {
                        "success": True,
                        "image_url": PNG_DATA_URL,
                        "file_path": "/tmp/generated.png",
                        "file_name": "generated.png",
                        "model": payload.get("model") or "gemini-3.1-flash-image-preview",
                        "prompt": payload.get("prompt") or "",
                        "workflow_type": "image_playground",
                        "images": images,
                        "batch_count": batch_count,
                        "success_count": batch_count,
                        "failed_count": 0,
                        "zip_path": "/outputs/e2e/image_playground/batch.zip",
                        "zip_file_name": "batch.zip",
                    }
                ),
            )
            return

        route.continue_()

    page.route("**/api/v1/**", handler)
    runtime_config_json = json.dumps(runtime_config)
    page.add_init_script(
        f"""
        window.localStorage.setItem('paper2any_e2e_bypass_auth', '1');
        window.localStorage.setItem('i18nextLng', 'en');
        window.localStorage.setItem('paper2any_runtime_config', {json.dumps(runtime_config_json)});
        """
    )


def test_image_playground_page_flow_with_playwright() -> None:
    captured: dict[str, object] = {}

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1440, "height": 1600})
        _install_mock_api(page, captured)

        page.goto(f"{BASE_URL}/image-playground", wait_until="domcontentloaded")
        page.wait_for_selector("text=Image Model Playground")
        page.wait_for_selector("text=Text Language")
        page.wait_for_selector("text=Batch Count")
        page.wait_for_selector("text=Aspect Ratio")
        page.wait_for_selector("text=Resolution")

        textareas = page.locator("textarea")
        textareas.nth(0).fill("Graph neural network for molecule property prediction.")
        page.get_by_role("button", name="Generate (2 points)").click()
        page.wait_for_selector("text=Please sign in to continue.")

        page.wait_for_function("() => Boolean(window.__PAPER2ANY_E2E__?.setMockUser)")
        page.evaluate(
            """
            () => window.__PAPER2ANY_E2E__.setMockUser({
              id: 'e2e-user',
              email: 'e2e@example.com',
            })
            """
        )

        page.get_by_role("button", name="Image 2 All").click()
        page.wait_for_selector("text=Image 2 All does not support adjustable aspect ratio")
        assert page.locator("select").count() == 2

        page.get_by_role("button", name=re.compile(r"^Image 2 OpenAI-style")).click()
        page.wait_for_selector("text=Size")
        page.wait_for_selector("text=Quality")
        assert page.locator("select").count() == 4
        page.locator("select").nth(2).select_option("1536x1024")
        page.locator("select").nth(3).select_option("high")

        page.get_by_role("button", name="Image 2 All").click()
        page.locator("select").nth(0).select_option("zh")
        page.locator("select").nth(1).select_option("4")
        page.get_by_role("button", name="Computer Science Paper Figure").click()
        page.get_by_role("button", name="Experiment Pipeline").click()
        textareas.nth(1).fill("Use a restrained blue palette with wide whitespace.")

        prompt_preview = page.locator("pre")
        assert "Graph neural network for molecule property prediction." in prompt_preview.text_content()

        page.get_by_role("button", name="Generate 4 Images (8 points)").click()
        page.wait_for_selector("text=The generated image has been saved to History Files")
        page.wait_for_selector("text=4 / 4 images generated")
        assert page.locator("img[alt='generated-1']").is_visible()
        assert page.get_by_role("link", name="Download All").is_visible()
        assert page.get_by_role("link", name="Download Image").count() == 4

        browser.close()

    assert int(captured["quota_calls"]) >= 1
    assert len(captured["generate_payloads"]) == 1

    request_payload = captured["generate_payloads"][0]
    request_headers = captured["generate_headers"][0]
    assert request_payload["model"] == "gpt-image-2-all"
    assert request_payload["template_key"] == "cs"
    assert request_payload["domain_key"] == "cs"
    assert request_payload["batch_count"] == 4
    assert "size" not in request_payload
    assert "quality" not in request_payload
    assert "computer-science research figure" in request_payload["prompt"]
    assert "pipeline-style composition" in request_payload["prompt"]
    assert "simplified Chinese" in request_payload["prompt"]
    assert "restrained blue palette" in request_payload["prompt"]
    assert request_headers["x-workflow-amount"] == "8"
