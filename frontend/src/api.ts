import axios from 'axios';
import type { GameMetadata, RecommendationRequest, ListResponse } from './types';

// In production, we serve from the same domain, so use relative paths.
// In development (Vite dev server), we fallback to port 8000.
export const API_BASE_URL = import.meta.env.VITE_API_URL || (import.meta.env.DEV ? 'http://127.0.0.1:8000' : '');

const api = axios.create({
  baseURL: API_BASE_URL,
});

export const getGenres = async (): Promise<string[]> => {
  const response = await api.get('/genres');
  return response.data;
};

export const getTags = async (): Promise<string[]> => {
  const response = await api.get('/tags');
  return response.data;
};

export const getTagDimensions = async (): Promise<Record<string, any>> => {
  const response = await api.get('/tag_dimensions');
  return response.data;
};

export const getTermLinks = async (): Promise<Record<string, string>> => {
  const response = await api.get('/term_links');
  return response.data;
};

export const getGames = async (): Promise<string[]> => {
  const response = await api.get('/games');
  return response.data;
};

export const searchGames = async (query: string): Promise<string[]> => {
  if (!query || query.length < 2) return [];
  const response = await api.get('/games/search', {
    params: { q: query, limit: 50 },
  });
  return response.data;
};

export const getRandomGame = async (): Promise<string> => {
  const response = await api.get('/games/random');
  return response.data;
};

export const getRandomTrendingGame = async (): Promise<string> => {
  const response = await api.get('/games/trending/random');
  return response.data;
};

export const getChangelog = async (): Promise<string> => {
  const response = await api.get('/changelog');
  return response.data.content;
};

export const getList = async (category: string, discoveryPref: number = 0): Promise<ListResponse> => {
  const response = await api.get(`/lists/${category}`, {
    params: { discovery_pref: discoveryPref },
  });
  return response.data;
};

export const getMetadata = async (names: string[]): Promise<GameMetadata[]> => {
  if (!names || names.length === 0) return [];
  const response = await api.post('/metadata', { names });
  return response.data;
};

export const recommend = async (request: RecommendationRequest): Promise<GameMetadata[]> => {
  const response = await api.post('/recommend', request);
  return response.data;
};

export default api;
