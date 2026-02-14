import axios from 'axios';
import type { GameMetadata, RecommendationRequest, ListResponse } from './types';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://127.0.0.1:8000';

const api = axios.create({
  baseURL: API_BASE_URL,
});

export const getGenres = async (): Promise<string[]> => {
  const response = await api.get('/genres');
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

export const getList = async (category: string, discoveryPref: number = 0): Promise<ListResponse> => {
  // Negate discoveryPref before submission as its meaning is reversed in the backend
  const response = await api.get(`/lists/${category}`, {
    params: { discovery_pref: -discoveryPref },
  });
  return response.data;
};

export const getMetadata = async (names: string[]): Promise<GameMetadata[]> => {
  if (!names || names.length === 0) return [];
  const response = await api.post('/metadata', { names });
  return response.data;
};

export const recommend = async (request: RecommendationRequest): Promise<GameMetadata[]> => {
  // Negate disc_pref before submission as its meaning is reversed in the backend
  const payload = {
    ...request,
    disc_pref: -request.disc_pref
  };
  const response = await api.post('/recommend', payload);
  return response.data;
};

export default api;
