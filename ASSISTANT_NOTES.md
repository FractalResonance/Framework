# Claude Assistant Notes

This file contains reference information to help Claude quickly get up to speed with the FRC Framework repository when accessing it in future sessions.

## Repository Structure

The repository is organized as follows:

- `/publications/` - Contains all FRC papers
  - `/publications/FRC_100.001/` - Foundational paper introducing the FRC framework
  - `/publications/FRC_100.002/` - Application paper focused on quantum chaos in stadium billiard

## Paper Publication Information

### FRC 100.001
- **Title**: Fractal Resonance Cognition: A Framework for Complex Systems Analysis
- **Author**: Hadi Servat
- **Zenodo DOI**: [10.5281/zenodo.15073056](https://zenodo.org/records/15073056)
- **License**: CC BY-NC-ND 4.0
- **Published**: March 23, 2025
- **Local Figures**: Located in `/Users/hadi/Development/FRC/Publish/100001/`

### FRC 100.002
- **Title**: Fractal Resonance Cognition in Quantum Chaos: Nodal Patterns in the Stadium Billiard
- **Author**: Hadi Servat
- **Zenodo DOI**: [10.5281/zenodo.15079278](https://zenodo.org/records/15079278)
- **License**: CC BY-NC-ND 4.0
- **Published**: March 24, 2025
- **Local Figures**: Located in `/Users/hadi/Development/FRC/Publish/100002/`

## Versioning System

FRC publications follow a structured versioning system:
- `FRC XXX.YYY` - Canonical version in this repository
- `FRC XXX.YYY.Z` - Variant for specific publication platform (Z=2 for viXra, Z=3 for Zenodo, etc.)

## Local Working Directories

Important directories on the local machine:
- `/Users/hadi/Development/FRC/Publish/` - Contains the figure files and working drafts
- `/Users/hadi/Development/FRC/Publish/pdf_conversion/` - Working directory for generating PDFs

## Key Actions Performed

1. Created clean manuscripts removing "Grok 3" coauthor references
2. Generated HTML files with MathJax for proper equation rendering
3. Set up Python scripts for generating figures
4. Published FRC 100.001 to Zenodo on March 23, 2025
5. Published FRC 100.002 to Zenodo on March 24, 2025

## Notes for Future Sessions

- When updating papers, always reference the Zenodo DOIs for proper citation
- Use the Markdown to HTML to PDF conversion workflow for creating publication PDFs
- Update the CITATION.cff file when new DOIs are assigned
- Maintain consistent CC BY-NC-ND 4.0 licensing across all publications