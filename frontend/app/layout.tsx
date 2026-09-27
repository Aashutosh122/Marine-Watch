import type { Metadata } from 'next';
import './globals.css';

export const metadata: Metadata = {
  title: 'Marine Watch | Ghost Net & Debris Detection Dashboard',
  description:
    'AI-powered Side-Scan Sonar analysis system for detecting marine debris, ghost nets, shipwrecks and man-made anomalies on the seafloor. Built for SIH 2026.',
  keywords: 'sonar, marine debris, ghost net, shipwreck detection, YOLOv8, underwater AI',
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <head>
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link rel="preconnect" href="https://fonts.gstatic.com" crossOrigin="anonymous" />
        <link
          href="https://fonts.googleapis.com/css2?family=Exo+2:wght@300;400;600;700;800&family=Inter:wght@300;400;500;600&family=JetBrains+Mono:wght@400;500&display=swap"
          rel="stylesheet"
        />
      </head>
      <body>{children}</body>
    </html>
  );
}
