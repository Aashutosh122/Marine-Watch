# Frontend Dashboard - Marine Debris Detection

This directory contains the Next.js web application for the Marine Debris Detection System. It provides an intuitive dashboard for users to upload Side-Scan Sonar images, view AI predictions overlaid with bounding boxes, and export structured anomaly reports.

## Overview

The frontend interacts with the FastAPI backend to visualize the results of the computer vision pipeline. Features include:
- Uploading sonar imagery logs.
- Real-time visualization of detected anomalies (ghost nets, shipwrecks, etc.).
- Dynamic confidence scoring filters.
- Data export to CSV/JSON formats.
- Map integration for georeferenced anomaly visualization.

## Setup & Run

### Requirements
- Node.js 18+ (Node 22.x recommended)
- `pnpm` package manager

### Installation

```bash
# Install pnpm if not already installed
npm install -g pnpm

# Install project dependencies
pnpm install
```

### Development Server

Run the development server locally:

```bash
pnpm dev
```

Open [http://localhost:3000](http://localhost:3000) in your browser to interact with the dashboard.

## Deployment

This frontend is designed for seamless deployment on Vercel. Ensure the **Root Directory** is set to `frontend` in your Vercel project settings.
