# Current arXiv paper

- `main.tex`: start here to edit or compile the paper.
- `sections/`: included main-text and appendix sections.
- `figures/` and `tables/`: the published assets.
- `authors.tex`, bibliographies, and `aistats2027.sty`: required LaTeX support files.
- `flat_oceans.pdf` and `arxiv.zip`: compiled paper and upload-ready source package.
- `experiments/`: all reproduction code, settings, and locally generated measurements. See its README for installation and reproduction.

To build the PDF and regenerate the ZIP, run `python experiments/scripts/build_paper.py` from this folder. Tectonic is required, or pass `--pdflatex` for TeX Live. Build intermediates stay in `.build/`; the ZIP includes only the manuscript's declared LaTeX dependencies, never experiment code or results.

New manuscript dependencies must be added to `experiments/configs/paper_files.json` so the build and submission ZIP include them.
