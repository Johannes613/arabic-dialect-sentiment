// Simple API client for backend communication

const getBaseUrl = () => {
  if (process.env.REACT_APP_API_URL) return process.env.REACT_APP_API_URL.replace(/\/$/, '');
  // Use relative path when served by FastAPI (static hosting)
  return '';
};

const jsonHeaders = { 'Content-Type': 'application/json' };

const safeFetch = async (path, options = {}, timeoutMs = 10000) => {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const res = await fetch(`${getBaseUrl()}${path}`, { ...options, signal: controller.signal });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    return await res.json();
  } finally {
    clearTimeout(timeout);
  }
};

export const checkHealth = () => safeFetch('/api/health');

export const analyzeSentiment = async (text, dialect = undefined, includeExplanation = false) => {
  const body = JSON.stringify({ text, dialect, include_explanation: includeExplanation });
  return safeFetch('/api/analyze', { method: 'POST', headers: jsonHeaders, body });
};

export const analyzeBatch = async (texts, dialect = undefined, includeExplanations = false) => {
  const body = JSON.stringify({ texts, dialect, include_explanations: includeExplanations });
  return safeFetch('/api/analyze/batch', { method: 'POST', headers: jsonHeaders, body });
};

export const preprocessText = async (text) => {
  const formData = new FormData();
  formData.append('text', text);
  const res = await fetch(`${getBaseUrl()}/api/preprocess`, { method: 'POST', body: formData });
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  return res.json();
};



