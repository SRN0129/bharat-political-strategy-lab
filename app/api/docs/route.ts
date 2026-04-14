import { NextResponse } from 'next/server';

export async function GET() {
  return NextResponse.json({
    product: 'Bharat Political Strategy Lab',
    version: '1.0.0',
    basePath: '/api',
    endpoints: [
      {
        path: '/api/health',
        method: 'GET',
        description: 'Platform health, dataset counts, free-tier diagnostics',
        request: null,
        response: {
          status: 'ok',
          now: 'ISO date',
          datasets: { scenarios: 'number', sentimentYears: 'number', topics: 'number', storedRuns: 'number' },
          freeTier: { paidApis: false, localStorage: true, cloudRequired: false }
        }
      },
      {
        path: '/api/catalog?page=1&pageSize=12&q=',
        method: 'GET',
        description: 'Paginated + searchable scenario catalog',
        request: { page: 'number', pageSize: 'number(5-50)', q: 'string optional' },
        response: { page: 'number', pageSize: 'number', total: 'number', totalPages: 'number', data: 'ScenarioInput[]' }
      },
      {
        path: '/api/history',
        method: 'GET',
        description: 'Historical sentiment index, topic sentiment, regional proxy',
        request: null,
        response: { yearly: 'array', topicWise: 'array', regionalProxy: 'array' }
      },
      {
        path: '/api/scenarios',
        method: 'GET',
        description: 'Stored simulation runs from local persistence',
        request: null,
        response: 'SimulationRun[]'
      },
      {
        path: '/api/simulate',
        method: 'POST',
        description: 'Run full simulation and persist run',
        request: {
          title: 'string',
          policyText: 'string',
          launchWindow: '12m|6m|3m|1m',
          regionFocus: 'national|north|south|east|west',
          visibility: 'number(0-100)',
          estimatedCost: 'number',
          beneficiariesM: 'number',
          manifestoText: 'string optional'
        },
        response: 'SimulationRun'
      },
      {
        path: '/api/manifesto',
        method: 'POST',
        description: 'Manifesto feasibility/contradiction analysis helper',
        request: { manifestoText: 'string' },
        response: { feasibilityScore: 'number', contradictions: 'string[]', optimized: 'string[]' }
      },
      {
        path: '/api/docs',
        method: 'GET',
        description: 'Machine-readable API index for product users',
        request: null,
        response: 'This document'
      }
    ]
  });
}
