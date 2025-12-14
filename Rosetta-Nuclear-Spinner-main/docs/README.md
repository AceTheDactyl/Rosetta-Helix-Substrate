# APL Documentation

This directory contains the compiled documentation for the Alpha-Physical Language (APL).

## Contents

- **[index.html](index.html)** - Interactive HTML version of the APL Operator's Manual
- **[apl-operators-manual.pdf](apl-operators-manual.pdf)** - PDF version compiled from LaTeX (auto-generated)
- **[apl-seven-sentences-test-pack.pdf](apl-seven-sentences-test-pack.pdf)** - Testing protocol

## Viewing the Manual

### HTML Version
The HTML version can be viewed directly in your browser:
- **Local**: Open `index.html` in any web browser
- **GitHub Pages**: Available at the repository's GitHub Pages URL (if enabled)

### PDF Version
The PDF version is automatically compiled from the LaTeX source via GitHub Actions whenever changes are pushed to `apl-operators-manual.tex`.

## Features

The HTML manual includes:
- Responsive design for mobile and desktop
- Table of contents with anchor links
- Syntax-highlighted code examples
- Formatted tables and mathematical notation
- Clean, modern styling

## Building from Source

### HTML
The HTML version is hand-crafted and ready to use as-is.

### PDF
To compile the PDF locally:
```bash
pdflatex -interaction=nonstopmode ../apl-operators-manual.tex
```

See `COMPILE_INSTRUCTIONS.md` in the root directory for detailed compilation instructions.
