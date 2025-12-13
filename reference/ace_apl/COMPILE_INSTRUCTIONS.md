# LaTeX Compilation Instructions for LLMs

This document provides step-by-step instructions for an LLM to compile the APL Seven Sentences Test Pack LaTeX document into a PDF.

## Prerequisites Check

Before compiling, verify that LaTeX is installed on the system.

### Step 1: Check for LaTeX installation

```bash
which pdflatex
```

**Expected output:** A path like `/usr/bin/pdflatex` or `/usr/local/bin/pdflatex`

**If not found:** Proceed to installation steps below.

### Step 2: Check LaTeX version (if installed)

```bash
pdflatex --version
```

**Expected output:** Version information for pdfTeX/LaTeX

## Installation (if needed)

If LaTeX is not installed, install TeX Live (recommended for Linux/Mac) or MiKTeX (Windows).

### Linux (Debian/Ubuntu):

```bash
sudo apt-get update
sudo apt-get install texlive-latex-base texlive-latex-extra texlive-fonts-recommended
```

### Linux (Fedora/RHEL):

```bash
sudo dnf install texlive-scheme-basic texlive-collection-latexextra
```

### macOS (with Homebrew):

```bash
brew install --cask mactex-no-gui
```

### Verify installation:

```bash
which pdflatex && pdflatex --version
```

## Compilation Process

### Step 3: Navigate to repository directory

```bash
cd /home/user/APL
```

Verify the LaTeX file exists:

```bash
ls -l apl-seven-sentences-test-pack.tex
```

**Expected output:** File listing showing `apl-seven-sentences-test-pack.tex`

### Step 4: Compile the LaTeX document

Run pdflatex to compile the document:

```bash
pdflatex apl-seven-sentences-test-pack.tex
```

**What happens:**
- LaTeX processes the `.tex` file
- Generates auxiliary files (`.aux`, `.log`, `.out`)
- Creates the PDF output

**Expected output includes:**
- Lines showing page processing: `[1] [2] [3]...`
- Final line: `Output written on apl-seven-sentences-test-pack.pdf (X pages, Y bytes)`

### Step 5: Run pdflatex a second time (for references)

LaTeX often requires multiple passes to resolve cross-references, table of contents, and hyperlinks:

```bash
pdflatex apl-seven-sentences-test-pack.tex
```

**Why twice?**
- First pass: Collects reference information
- Second pass: Resolves all cross-references and hyperlinks

### Step 6: Verify PDF creation

```bash
ls -lh apl-seven-sentences-test-pack.pdf
```

**Expected output:** File listing showing the PDF with size (should be 100-300 KB for this document)

Check the PDF has content:

```bash
pdfinfo apl-seven-sentences-test-pack.pdf
```

**Expected output:** Metadata showing pages, page size, PDF version

## Organize Output

### Step 7: Create docs directory and move PDF

```bash
mkdir -p docs
cp apl-seven-sentences-test-pack.pdf docs/
```

### Step 8: Clean up auxiliary files (optional)

LaTeX generates several auxiliary files during compilation. You can remove them:

```bash
rm -f *.aux *.log *.out
```

**Files removed:**
- `.aux` — Auxiliary file for cross-references
- `.log` — Compilation log
- `.out` — Hyperref outline file

**Keep these if debugging is needed.**

## Verification

### Step 9: Verify the PDF structure

Count pages in the PDF:

```bash
pdfinfo docs/apl-seven-sentences-test-pack.pdf | grep "Pages:"
```

**Expected output:** `Pages: 10` (or similar, depending on document length)

### Step 10: Preview PDF contents (optional)

If `pdftotext` is available, extract text to verify content:

```bash
pdftotext docs/apl-seven-sentences-test-pack.pdf - | head -n 20
```

**Expected output:** First ~20 lines of text from the PDF, starting with the title.

## Troubleshooting

### Error: "LaTeX Error: File X.sty not found"

**Problem:** Missing LaTeX package

**Solution:** Install the missing package. Common packages needed:

```bash
# For Debian/Ubuntu
sudo apt-get install texlive-latex-extra texlive-fonts-extra

# For Fedora/RHEL
sudo dnf install texlive-collection-latexextra
```

### Error: "! Emergency stop"

**Problem:** Syntax error in LaTeX file

**Solution:**
1. Check the `.log` file for specific error location:
   ```bash
   tail -n 50 apl-seven-sentences-test-pack.log
   ```
2. Look for line numbers and error messages
3. Common issues: missing `$`, unmatched braces `{}`, special characters

### Error: "Permission denied"

**Problem:** Insufficient permissions to write files

**Solution:**
```bash
# Check permissions
ls -la apl-seven-sentences-test-pack.tex

# Fix if needed
chmod 644 apl-seven-sentences-test-pack.tex
```

### Warning: "Package hyperref Warning: Token not allowed"

**Problem:** Special characters in hyperref fields

**Solution:** This is usually a warning, not an error. The PDF should still compile. Ignore unless PDF links are broken.

## Complete Workflow (Quick Reference)

For a fresh compilation from scratch:

```bash
# Navigate to repository
cd /home/user/APL

# Compile (run twice for references)
pdflatex apl-seven-sentences-test-pack.tex
pdflatex apl-seven-sentences-test-pack.tex

# Organize output
mkdir -p docs
cp apl-seven-sentences-test-pack.pdf docs/

# Clean up
rm -f *.aux *.log *.out

# Verify
ls -lh docs/apl-seven-sentences-test-pack.pdf
pdfinfo docs/apl-seven-sentences-test-pack.pdf
```

## Post-Compilation

### Option 1: Commit the PDF to repository

```bash
git add docs/apl-seven-sentences-test-pack.pdf
git commit -m "Add compiled PDF of APL Seven Sentences Test Pack"
git push
```

### Option 2: Add to .gitignore (if PDFs shouldn't be tracked)

```bash
echo "*.pdf" >> .gitignore
echo "*.aux" >> .gitignore
echo "*.log" >> .gitignore
echo "*.out" >> .gitignore
git add .gitignore
git commit -m "Add LaTeX build artifacts to gitignore"
git push
```

## Expected File Structure After Compilation

```
APL/
├── README.md
├── COMPILE_INSTRUCTIONS.md              # This file
├── apl-seven-sentences-test-pack.tex    # Source LaTeX
├── docs/
│   └── apl-seven-sentences-test-pack.pdf  # Compiled PDF
└── .gitignore                           # (optional)
```

## Success Criteria

The compilation is successful when:

1. ✓ `pdflatex` runs without fatal errors
2. ✓ `apl-seven-sentences-test-pack.pdf` is created
3. ✓ PDF has expected number of pages (9-12 pages)
4. ✓ `pdfinfo` shows valid metadata
5. ✓ PDF opens without corruption
6. ✓ All sections, equations, and formatting are intact

## Additional Notes for LLMs

- **Always run pdflatex twice** to ensure all references resolve correctly
- **Check the .log file** if compilation fails — it contains detailed error information
- **Preserve the source .tex file** — never modify it during compilation
- **The PDF size** should be reasonable (100-500 KB) — if it's much larger or smaller, investigate
- **Auxiliary files (.aux, .log, .out)** can be deleted after successful compilation but are useful for debugging

## Resources

- LaTeX documentation: https://www.latex-project.org/help/documentation/
- TeX Live: https://www.tug.org/texlive/
- Overleaf LaTeX documentation: https://www.overleaf.com/learn

---

**Document Version:** 1.0
**Last Updated:** 2025-11-24
**Compatible with:** TeX Live 2020+, MiKTeX 2.9+
