# Project Docs

This folder holds additional documentation assets such as screenshots and diagrams.

## Images
- Put UI screenshots under `docs/images/` with these suggested names:
  - `ui-home.png` – home page of the Gradio app
  - `prediction.png` – prediction result (probability text + one image result)
  - `gradcam.png` – Grad-CAM heatmap or side-by-side visualization

## How to capture screenshots
1. Start the web UI:
   ```bash
   docker compose up --build web
   ```
2. Open http://localhost:7860 in your browser.
3. Upload a sample lesion image (place any test images under `data/` locally if you want a consistent set).
4. Take screenshots for the three views listed above and save them in `docs/images/`.

## Notes
- Screenshots are for illustration only; this project is not a medical device.
- If you add more pages/components to the app, include additional screenshots and reference them in the root `README.md`.
