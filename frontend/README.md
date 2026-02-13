# Modern React Frontend

This directory contains the modern, high-performance frontend for SteamJackalope.

## Tech Stack

- **Framework:** React 19
- **Build Tool:** Vite
- **Language:** TypeScript
- **Styling:** Tailwind CSS v4
- **Icons:** Lucide React
- **Animation:** Framer Motion
- **Markdown:** React Markdown + Remark GFM + Remark Math + Rehype KaTeX

## Features

- **Responsive Design:** Fully optimized for desktop and mobile devices.
- **Real-time Discovery:** Recommendations update as you tune sliders (debounced).
- **Hybrid Search:** Combine natural language prompts with multiple seed games.
- **Interactive Visualizations:** "Visualize Contributions" toggle shows how each factor affects a game's score.
- **Advanced Filtering:** Multi-selection genres, VR, NSFW blurring, unreleased games, etc.
- **Analysis View:** Curated lists of Steam extremes (Difficulty, Quality, Playtime, etc.).
- **Markdown Content:** About and Methodology pages rendered directly from project docs.

## Getting Started

### Installation

```bash
npm install
```

### Development

```bash
npm run dev
```
The app will be available at `http://127.0.0.1:3000`.

### Production Build

```bash
npm run build
```
The output will be in the `dist/` directory.

## Project Structure

- `src/api.ts`: API client for interacting with the FastAPI backend.
- `src/types.ts`: TypeScript interfaces for games, recommendations, and parameters.
- `src/components/`: Modular React components (GameCard, Filters, RecommendationsView, etc.).
- `src/App.tsx`: Main application logic and state management.
- `public/assets/`: Static assets (diagrams, icons).

## Networking

The frontend is configured to connect to the backend at `127.0.0.1:8000`. Using `127.0.0.1` instead of `localhost` is preferred on Windows to avoid DNS lookup delays and potential connection issues.
