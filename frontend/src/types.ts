export interface GameMetadata {
  appid: number;
  name: string;
  release_date: string;
  short_description: string;
  header_image?: string;
  release_year: number;
  estimated_playtime: number;
  difficulty_predicted: number;
  positive: number;
  negative: number;
  genres: string;
  tags: string;
  price?: string;
  is_free?: boolean;
  is_in_library?: boolean;
  is_nsfw?: boolean;
  is_delisted?: boolean;
  raw_pop?: number;
  raw_length?: number;
  weighted_score?: number;
  semantic_match?: number;
  tag_match?: number;
  rating?: number;
  personalized_quality?: number;
  match_percent?: number;
  is_personalized?: boolean;
  
  // Debug components
  z_semantic?: number;
  w_semantic?: number;
  z_tag?: number;
  w_tag?: number;
  z_spps?: number;
  w_spps?: number;
  z_date?: number;
  w_date?: number;
  z_pop?: number;
  w_pop?: number;
  z_length?: number;
  w_length?: number;
  z_difficulty?: number;
  w_difficulty?: number;
  z_price?: number;
  w_price?: number;

  // Fields for lists
  quality_score?: number;
  total_reviews?: number;
  playtime?: number;
  term_links?: Record<string, string>;
}

export interface RecommendationRequest {
  alpha: number;
  beta: number;
  quality_pref: number;
  age_pref: number;
  pop_pref: number;
  disc_pref: number;
  length_pref: number;
  difficulty_pref: number;
  price_pref: number;
  remove_vr: boolean;
  english_only: boolean;
  remove_nsfw: boolean;
  remove_utilities: boolean;
  remove_unreleased: boolean;
  remove_delisted: boolean;
  top_k: number;
  prompt: string;
  seed_games: string[];
  genres: string[];
  tags: string[];
  vibe_vector?: number[];
  semantic_vibe_vector?: number[];
  topic_vibe_vector?: number[];
  gamma_topic?: number;
  metadata_weights?: Record<string, number>;
  intercept?: number;
  scaling_factor?: number;
  
  // Profile Exclusion
  profile_filter?: 'none' | 'rated' | 'all';
  library_appids?: number[];
  rated_appids?: number[];
  ignored_appids?: number[];
  library_details?: Record<number, { playtime: number; personalized_q: number; p_plus_t: number }>;
}

export interface ListResponse {
  top: Partial<GameMetadata>[];
  bottom: Partial<GameMetadata>[];
  tag_impacts?: { tag: string; impact: number }[];
}
