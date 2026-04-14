import { NextResponse } from 'next/server';
import { buildSentimentIndex } from '@/lib/simulationEngine';

export async function GET() {
  return NextResponse.json(buildSentimentIndex());
}
