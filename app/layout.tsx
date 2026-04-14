import type { Metadata } from 'next';
import './globals.css';

export const metadata: Metadata = {
  title: 'Bharat Political Strategy Lab',
  description: 'AI-powered political strategy and policy simulation for India'
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className="min-h-screen">{children}</body>
    </html>
  );
}
