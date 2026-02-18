import type { Metadata } from 'next';
import './globals.css';

export const metadata: Metadata = {
    title: 'TradePulse — AI-Powered Options Trading Platform',
    description: 'Multi-user NIFTY options trading platform with UT Bot signals, real-time dashboard, and automated trading.',
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
    return (
        <html lang="en">
            <body>{children}</body>
        </html>
    );
}
