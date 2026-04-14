import { NextResponse } from 'next/server';
import { readRuns } from '@/lib/storage';

export async function GET() {
  return NextResponse.json(readRuns());
}
