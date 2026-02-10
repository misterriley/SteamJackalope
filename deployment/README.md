# Deployment Guides

This directory contains scripts and documentation for deploying SteamJackalope to various hosting providers.

## Contents

- `RACKNERD.md` - Step-by-step guide for deploying on RackNerd VPS
- `measure_memory.py` - Memory footprint measurement tool
- `measure_memory.bat` - Windows batch file to run memory tests
- `test_*.py` - Memory testing scripts for different scenarios

## Quick Start

For RackNerd deployment:
1. Follow the guide in `RACKNERD.md`
2. Use `measure_memory.py` to verify memory usage after deployment

## Memory Requirements

- **Backend only (no model)**: ~180 MB
- **Backend + Frontend (no model)**: ~676 MB
- **Backend + Frontend (with model loaded)**: ~890 MB
- **Recommended VPS**: 2 GB RAM minimum

## Notes

- The app uses lazy loading for the SentenceTransformer model (only loads when user enters a prompt)
- All large arrays are memory-mapped to keep RAM usage low
- ONNX backend is used for the transformer model (minimal memory impact)