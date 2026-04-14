import { NextResponse } from 'next/server';
import scenarios from '@/data/sample_scenarios.json';
import sentiment from '@/data/sentiment_2000_2025.json';
import topics from '@/data/topic_sentiment.json';
import { readRuns } from '@/lib/storage';

export async function GET() {
  const runs = readRuns();
  return NextResponse.json({
    status: 'ok',
    now: new Date().toISOString(),
    datasets: {
      scenarios: scenarios.length,
      sentimentYears: sentiment.length,
      topics: topics.length,
      storedRuns: runs.length
    },
    freeTier: {
      paidApis: false,
      localStorage: true,
      cloudRequired: false
    }
  });
}
