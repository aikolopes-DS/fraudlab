"""Exercise the built case study in Chromium; writes local QA screenshots."""

from __future__ import annotations

import argparse
import functools
import json
import threading
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from playwright.sync_api import sync_playwright

SITE_DIR = Path(__file__).resolve().parent


class QuietHandler(SimpleHTTPRequestHandler):
    def log_message(self, *_args):
        pass


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--screenshots", type=Path, default=SITE_DIR.parent / "artifacts" / "qa" / "site")
    args = parser.parse_args()
    args.screenshots.mkdir(parents=True, exist_ok=True)
    handler = functools.partial(QuietHandler, directory=str(SITE_DIR / "dist"))
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_port}"
    evidence = json.loads((SITE_DIR / "dist" / "evidence.json").read_text(encoding="utf-8"))
    try:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch()
            errors = []
            checks = []
            for label, viewport in (("desktop", {"width": 1440, "height": 1050}), ("mobile", {"width": 390, "height": 844})):
                page = browser.new_page(viewport=viewport, reduced_motion="reduce")
                page.on("pageerror", lambda error: errors.append(str(error)))
                page.goto(base, wait_until="networkidle")
                page.wait_for_function("document.querySelector('[data-stage]').textContent !== 'Carregando evidências'")
                assert page.locator("h1").is_visible(), "Hero must be visible"
                assert not page.locator("[data-load-error]").is_visible(), "Evidence must load"
                assert page.evaluate("document.documentElement.scrollWidth <= innerWidth"), f"Horizontal overflow at {label} width"
                displayed = {key: page.locator(selector).inner_text() for key, selector in (("local_auc", "[data-local-auc]"), ("public_auc", "[data-public-auc]"), ("ranking", "[data-ranking]"))}
                if evidence.get("checks", {}).get("training_verified") and evidence.get("results", {}).get("local_auc") is not None:
                    assert displayed["local_auc"] == f"{evidence['results']['local_auc']:.5f}".replace(".", ",")
                if evidence.get("checks", {}).get("submission_verified") and evidence.get("results", {}).get("public_auc") is not None:
                    assert displayed["public_auc"] == f"{evidence['results']['public_auc']:.5f}".replace(".", ",")
                    assert displayed["ranking"] == f"{evidence['results']['ranking']:,}º".replace(",", ".")
                assert page.locator(".metric-card").count() == 3
                weights = evidence.get("run", {}).get("effective_weights", {})
                if weights:
                    assert page.locator("[data-composition]").is_visible()
                    composition = page.locator("[data-composition-weights]").inner_text().replace("\u00a0", "")
                    for weight in weights.values():
                        percent = f"{weight * 100:.2f}".rstrip("0").rstrip(".").replace(".", ",") + "%"
                        assert percent in composition
                    attribution = page.locator("[data-component-credits]").inner_text()
                    for component in evidence.get("external_components", []):
                        assert component["trained_by"] in attribution
                        assert component["predictions_license"] in attribution
                        assert component["code_license"] in attribution
                    if evidence.get("run", {}).get("phase") == "exploratory_full_data_cv":
                        assert page.locator("[data-independent-note]").is_visible()
                        assert not page.locator("[data-holdout]").is_visible()
                page.locator('nav a[href="#resultados"]').click()
                assert "#resultados" in page.url, "Results navigation must work"
                page.locator('nav a[href="#materiais"]').click()
                assert "#materiais" in page.url, "Materials navigation must work"
                page.goto(base, wait_until="networkidle")
                page.screenshot(path=str(args.screenshots / f"{label}.png"), full_page=True)
                checks.append({"viewport": label, "overflow": False, "evidence_loaded": True, "navigation": True, "displayed_results": displayed, "effective_weights": weights, "external_authorship_visible": bool(weights)})
                page.close()

            # Separate browser-only fixtures verify labels and trust boundaries.
            page = browser.new_page()
            fixture = {"checks": {"training_verified": True, "submission_verified": False}, "results": {"local_auc": 0.91, "fold_scores": [0.90, 0.92], "public_auc": 0.99, "ranking": 1}, "links": {"video": "javascript:alert(1)"}, "updated_at": "2026-11-16", "run": {"model": "reference", "validation": {"method": "stratified", "n_splits": 2}}}
            page.route("**/evidence.json", lambda route: route.fulfill(json=fixture))
            page.goto(base, wait_until="networkidle")
            assert page.locator("[data-local-auc]").inner_text() == "0,91000"
            assert page.locator("[data-public-auc]").inner_text() == "—"
            assert page.locator("[data-ranking]").inner_text() == "—"
            assert page.locator("[data-local-context]").inner_text() == "reference · stratified · 2 folds"
            assert page.locator("[data-updated]").inner_text().endswith("16/11/2026")
            assert page.locator("[data-fold-panel]").is_visible()
            assert not page.locator("[data-video-container]").is_visible()
            assert errors == [], f"Browser errors: {errors}"
            checks.append({"test_fixture_only": True, "unverified_results_hidden": True, "unsafe_video_link_hidden": True, "date_only_preserved": True})
            browser.close()
            (args.screenshots / "checks.json").write_text(json.dumps({"scope": "Local rendering of the current public evidence. UI fixtures are separate checks, not competition results.", "observed_at": evidence.get("results", {}).get("observed_at"), "checks": checks, "page_errors": errors}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            print("Browser checks passed: desktop, mobile, evidence rendering, and navigation.")
    finally:
        server.shutdown()
        server.server_close()


if __name__ == "__main__":
    main()
