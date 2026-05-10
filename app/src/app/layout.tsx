import type { Metadata, Viewport } from 'next';
import './globals.css';
import PolicyEngineHeader from '@/components/PolicyEngineHeader';

const TITLE = 'UK Salary Sacrifice Cap Analysis';
const DESCRIPTION =
  'Interactive analysis of the UK Autumn Budget 2025 NI-exempt salary-sacrifice cap. Explore revenue and distributional outcomes across cap levels and behavioural scenarios.';

export const metadata: Metadata = {
  title: TITLE,
  description: DESCRIPTION,
};

export const viewport: Viewport = {
  width: 'device-width',
  initialScale: 1.0,
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <head>
        <link
          rel="stylesheet"
          href="https://unpkg.com/@policyengine/design-system/dist/tokens.css"
        />
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link
          rel="preconnect"
          href="https://fonts.gstatic.com"
          crossOrigin="anonymous"
        />
        <link
          href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap"
          rel="stylesheet"
        />
      </head>
      <body>
        <PolicyEngineHeader />
        {children}
      </body>
    </html>
  );
}
