# SalsaAgent Project Page

Static project webpage for **SalsaAgent: A Multimodal Embodied Language Model for Interactive Dance Generation**.

## Structure

- `index.html` — main project page
- `style.css` — theme and example slideshow styles
- `asset/` — media and qualitative comparison clips
  - `Salsa_Agent_Demo.mp4` — overview video
  - `teaser.png`, `framework.png`, `subjective_chart.png` — figures from the paper
  - `examples.json` — metadata for the comparison slideshow
  - `Examples/` — per-clip videos (Ground Truth, SalsaAgent, Duolando, InterGen)

## Local preview

From this directory:

```bash
python3 -m http.server 8080
```

Then open `http://localhost:8080/` in a browser.

## Deployment

Host the contents of `docs/` on GitHub Pages (project site root or `/docs` folder), or any static file server. Large `.mp4` files may require Git LFS or external hosting if the repository size limit is a concern.

## Updating

- **Paper link**: set the `href` on `#paper-link` in `index.html` when the arXiv / IEEE URL is available.
- **New examples**: add folders under `asset/Examples/` and regenerate or edit `asset/examples.json`.
- **Authors**: edit the author block in `index.html` and the BibTeX entry if needed.
