import numpy as np
import re
import os
import pandas as pd
from common.constants import (
    EPSILON, Z_SCORE_CLAMP_MIN, Z_SCORE_CLAMP_MAX, SOFTMIN_TEMPERATURE,
    KERNEL_VETO_PENALTY, KERNEL_RESCUE_THRESHOLD, KERNEL_SOUL_MATCH_THRESHOLD,
    KERNEL_MOOD_CLASH_PENALTY, KERNEL_CINEMATIC_BOOST, KERNEL_CRPG_BOOST,
    KERNEL_SOFT_GATE_TEMP, HORROR_MARKERS, HARD_ANCHORS, SYMMETRIC_ANCHORS,
    SERIOUS_TAGS, CUTE_TAGS, SERIOUS_MOOD_TAGS, LIGHT_MOOD_TAGS, NARRATIVE_TAGS,
    KERNEL_IDENTITY_POWER, KERNEL_THEMATIC_CLASH_PENALTY,
    KERNEL_HARD_POLLUTION_PENALTY, KERNEL_SOFT_POLLUTION_PENALTY,
    KERNEL_STRONG_NSFW_PENALTY, KERNEL_MATURE_NSFW_PENALTY,
    KERNEL_VR_PENALTY, KERNEL_PERSPECTIVE_PENALTY
)

# --- Mechanical Identity Groups (MIGs) ---
MIGS = {
    "SHOOTER": {"FPS", "Arena Shooter", "Boomer Shooter", "Hero Shooter", "Tactical FPS", "Third-Person Shooter", "Shooter"},
    "MELEE_ACTION": {"Souls-like", "Spectacle fighter", "Hack and Slash", "Character Action Game", "Swordplay"},
    "ACTION_ADVENTURE": {"Action-Adventure", "Action"},
    "PUZZLE_LOGIC": {"Logic", "Sokoban", "Programming", "Coding", "Automation", "Word Game", "Puzzle"},
    "PUZZLE_SPATIAL": {"Puzzle", "Puzzle Platformer", "3D Platformer", "Hidden Object"},
    "PLATFORMER": {"Precision Platformer", "Runner", "Platformer"},
    "STRATEGY": {"Turn-Based Strategy", "Turn-Based Tactics", "RTS", "Real-Time", "Action RTS", "Tower Defense", "4X"},
    "MANAGEMENT": {"Colony Sim", "City Builder", "Management", "Shop Keeper", "Economy", "Resource Management", "Inventory Management"},
    "BUILDING": {"Building", "Base Building", "Sandbox"},
    "SURVIVAL": {"Survival", "Survival Horror", "Open World Survival Craft"},
    "ROGUELIKE": {"Roguelike", "Roguelite", "Action Roguelike", "Traditional Roguelike", "Roguelike Deckbuilder"},
    "NARRATIVE_VN": {"Visual Novel", "Anime", "Otome"},
    "DATING_SIM": {"Dating Sim"},
    "NARRATIVE_STORY": {"Story Rich", "Choices Matter", "Multiple Endings", "Cinematic", "Emotional", "Interactive Fiction", "Romance", "Quick Time Events"},
    "RPG_TRADITIONAL": {"Party-Based RPG", "Dungeon Crawler", "Turn-Based Combat", "Tactical RPG"},
    "RPG_MODERN": {"Action RPG", "JRPG", "RPG"},
    "CRPG_IDENTITY": {"CRPG", "Isometric"},
    "POINT_AND_CLICK": {"Point & Click"},
    "DETECTIVE": {"Detective", "Investigation", "Mystery"},
    "CASUAL": {"Casual", "Relaxing", "Family Friendly", "Colorful", "Cartoony"},
    "HORROR": {"Horror", "Survival Horror", "Psychological Horror", "Dark"},
    "METROIDVANIA": {"Metroidvania", "Roguevania"},
    "VR": {"VR", "VR Only"},
    "VEHICLE_SIM": {"Flight Sim", "Space Sim", "Racing", "Automobile Sim", "Flight"},
    "SIMULATION": {"Simulation", "Life Sim", "Farming Sim", "Medical Sim", "Job Simulator"},
    "OPEN_WORLD": {"Open World", "Sandbox", "Open World Survival Craft"},
    "HENTAI": {"Hentai", "NSFW"},
    "NUDITY": {"Nudity", "Sexual Content"},
    "FMV": {"FMV", "Live-action"},
    "SCI_FI": {"Sci-fi", "Futuristic", "Space", "Space Sim"},
    "CYBERPUNK": {"Cyberpunk"},
    "FANTASY": {"Fantasy", "Magic", "Medieval", "Dark Fantasy"},
    "HISTORICAL": {"Historical", "World War II", "World War I", "History"},
    "SURREAL": {"Surreal", "Psychedelic", "Stylized", "Experimental"}
}

# --- MIG Taxonomy (v6.0 Oracle) ---
STRUCTURAL_MIGS = {"STRATEGY", "ROGUELIKE", "MANAGEMENT", "BUILDING", "SURVIVAL", "RPG_TRADITIONAL", "VEHICLE_SIM", "METROIDVANIA", "VR", "FMV", "HENTAI"}
COGNITIVE_MIGS = {"DETECTIVE", "PUZZLE_LOGIC", "NARRATIVE_STORY"}
SEMI_STRUCTURAL_MIGS = {"SHOOTER", "RPG_MODERN", "MELEE_ACTION", "CRPG_IDENTITY"}

MIG_WEIGHTS = {
    "STRATEGY": 15.0, "ROGUELIKE": 15.0, "MANAGEMENT": 12.0, "BUILDING": 12.0, "SURVIVAL": 12.0,
    "VEHICLE_SIM": 12.0, "METROIDVANIA": 10.0, "VR": 15.0, "FMV": 10.0, "HENTAI": 15.0,
    "DETECTIVE": 15.0, "PUZZLE_LOGIC": 15.0, "NARRATIVE_STORY": 8.0,
    "CYBERPUNK": 4.0, "NARRATIVE_VN": 8.0, "DATING_SIM": 8.0, "CRPG_IDENTITY": 8.0, 
    "POINT_AND_CLICK": 5.0, "SHOOTER": 6.0, "RPG_TRADITIONAL": 6.0, 
    "RPG_MODERN": 6.0, "HORROR": 4.0, "MELEE_ACTION": 6.0,
    "PUZZLE_SPATIAL": 3.0, "PLATFORMER": 3.0,
    "ACTION_ADVENTURE": 0.1, "SIMULATION": 0.1, "OPEN_WORLD": 0.1, "CASUAL": 0.1
}

def apply_thematic_clash(kernel, candidate_masks, seed_migs, is_2d=False, identity_match=None, vibe_shield=None):
    if not seed_migs: return kernel
    id_boost = np.where(identity_match > 0.4, 2.0, 1.0) if identity_match is not None else 1.0
    def get_clash_penalty(seed_key, candidate_key, is_2d_internal):
        m_cand = np.zeros(kernel.shape[0], dtype=bool)
        for t in MIGS[candidate_key]:
            if t in candidate_masks: m_cand |= candidate_masks[t]
        p = 1.0 - (1.0 - KERNEL_THEMATIC_CLASH_PENALTY) / id_boost
        if not is_2d_internal:
            if seed_key in seed_migs and candidate_key not in seed_migs: 
                v = p + (1.0 - p) * (~m_cand).astype(float)
                # Setting exemption for top 1% vibe
                return np.where(vibe_shield if vibe_shield is not None else False, 1.0, v)
            return 1.0
        else:
            seed_has_primary = np.array([seed_key in m and candidate_key not in m for m in seed_migs])
            v = np.where(seed_has_primary, p + (1.0 - p) * (~m_cand)[:, None], 1.0)
            return np.where(vibe_shield[:, None] if vibe_shield is not None else False, 1.0, v)
    kernel *= get_clash_penalty("SCI_FI", "FANTASY", is_2d)
    kernel *= get_clash_penalty("FANTASY", "SCI_FI", is_2d)
    kernel *= get_clash_penalty("HISTORICAL", "FANTASY", is_2d)
    return kernel

def apply_kernel_vetoes(kernel, candidate_masks, seed_tags, seed_migs, is_2d=False, identity_match=None, vibe_shield=None, soul_match_mask=None):
    if candidate_masks is None: return kernel
    veto_multiplier = np.ones_like(kernel, dtype=np.float32)
    id_boost = np.where(identity_match > 0.4, 2.0, 1.0) if identity_match is not None else 1.0
    def get_softened_penalty(penalty, boost_arr): return 1.0 - (1.0 - penalty) / boost_arr
    perspectives_2d, perspectives_3d = ["2D", "Side Scroller"], ["3D", "Third Person", "First-Person", "Isometric", "Third-Person Shooter", "FPS", "Quick Time Events", "Cinematic"]
    if not is_2d:
        is_2d_seed, is_3d_seed = any(p in (seed_tags or []) for p in perspectives_2d), any(p in (seed_tags or []) for p in perspectives_3d)
        m_2d, m_3d = np.zeros(len(kernel), dtype=bool), np.zeros(len(kernel), dtype=bool)
        for p in perspectives_2d:
            if p in candidate_masks: m_2d |= candidate_masks[p]
        for p in perspectives_3d:
            if p in candidate_masks: m_3d |= candidate_masks[p]
        p_penalty = get_softened_penalty(KERNEL_PERSPECTIVE_PENALTY, id_boost * 1.5)
        # Oracle Override: If intellectual soul matches,camera doesn't matter
        if soul_match_mask is not None: p_penalty = np.where(soul_match_mask, 1.0, p_penalty)
        v = np.ones_like(kernel)
        if is_2d_seed and not is_3d_seed: v = (p_penalty + (1.0 - p_penalty) * m_2d.astype(float))
        elif is_3d_seed and not is_2d_seed: v = (p_penalty + (1.0 - p_penalty) * m_3d.astype(float))
        veto_multiplier *= np.where(vibe_shield if vibe_shield is not None else False, 1.0, v)
    else:
        # 2D case simplified for Oracle
        veto_multiplier *= 1.0
    return kernel * veto_multiplier

def apply_identity_protection(kernel, candidate_masks, seed_tags, seed_migs, is_2d=False, identity_match=None, vibe_shield=None):
    if not seed_migs: return kernel
    id_boost = np.where(identity_match > 0.4, 2.0, 1.0) if identity_match is not None else 1.0
    def get_group_penalty(group, penalty, is_2d_internal):
        m = np.zeros(kernel.shape[0], dtype=bool)
        for t in MIGS[group]:
            if t in candidate_masks: m |= candidate_masks[t]
        p = 1.0 - (1.0 - penalty) / id_boost
        if not is_2d_internal:
            seed_has = group in seed_migs or any(t in (seed_tags or []) for t in MIGS[group])
            v = p + (1.0 - p) * (~m).astype(float) if not seed_has else 1.0
            return np.where(vibe_shield if vibe_shield is not None else False, 1.0, v)
        else: return 1.0
    for group in STRUCTURAL_MIGS: kernel *= get_group_penalty(group, 0.05, is_2d)
    for group in SEMI_STRUCTURAL_MIGS: kernel *= get_group_penalty(group, 0.4, is_2d)
    return kernel

# --- Structural Identity Constants ---
CEREBRAL_DIMENSIONS = {5, 6, 12, 20, 37, 40, 69, 108, 118, 123, 149, 193, 216} # Key intellectual/mystery dimensions
TROPE_DIMENSIONS = {0, 6, 12, 56, 65, 102, 133, 160, 161, 165, 171, 179, 185} # Dimensions containing common noise/tropes

def calculate_behavioral_resonance(embeddings_graph, seed_graph_vec):
    """Calculates similarity in the 128-dimensional latent behavioral space."""
    if embeddings_graph is None or seed_graph_vec is None: return 1.0
    dot = np.dot(embeddings_graph.astype(np.float32), seed_graph_vec.astype(np.float32))
    norms = np.linalg.norm(embeddings_graph, axis=1) * np.linalg.norm(seed_graph_vec) + 1e-9
    return dot / norms

def fast_jsd_similarity(P, Q, mean=0.0, std=1.0):
    """Calculates Jensen-Shannon similarity between two probability distributions."""
    eps = 1e-10
    M = 0.5 * (P + Q + eps)
    js_div = 0.5 * (np.sum(P * np.log((P + eps) / M), axis=-1) + np.sum(Q * np.log((Q + eps) / M), axis=-1))
    dist = np.sqrt(np.maximum(js_div, 0))
    sim = 1.0 - dist
    return (sim - mean) / (std + 1e-9)

def calculate_jackalope_kernel(
    verb_profiles, seed_verb_profile, sem_vectors, sem_norms, seed_sem_vec, seed_sem_norm, topic_distributions, seed_topic_dist, topic_means, topic_stds, tag_scaling_factor, dot_product_lambda, sem_scaling_factor, sem_lambda, topic_scaling_factor=0.1, mature_content_flags=None, seed_mature_content=False, seed_migs=None, seed_tags=None, candidate_anchor_masks=None, active_narrative_seed=None, is_cinematic_seed=False, is_crpg_seed=False, precalculated_masks=None, difficulty_z=None, seed_difficulty_z=None, tone_z=None, seed_tone_z=None, temperature=0.01, return_components=False, graph_embeddings=None, seed_graph_vec=None
):
    import pandas as pd
    diff_sim = np.exp(-0.5 * ((difficulty_z.astype(np.float32) - float(seed_difficulty_z or 0)) / 3.0)**2) if difficulty_z is not None else 1.0
    tone_sim = np.exp(-0.5 * ((tone_z.astype(np.float32) - float(seed_tone_z or 0)) / 3.0)**2) if tone_z is not None else 1.0
    sem_sims = (np.dot(sem_vectors.astype(np.float32), seed_sem_vec.astype(np.float32)) / (sem_norms + sem_lambda)) / (seed_sem_norm + sem_lambda)
    
    # 1. Identity Signal
    active_seed_migs, identity_match = set(seed_migs or []), np.ones(len(verb_profiles), dtype=np.float32)
    soul_match_mask = np.zeros(len(verb_profiles), dtype=bool)
    structural_seed_match = any(g in STRUCTURAL_MIGS for g in active_seed_migs)
    if candidate_anchor_masks:
        inter_w, union_w = np.zeros(len(verb_profiles), dtype=np.float32), np.zeros(len(verb_profiles), dtype=np.float32)
        seed_cog_tags = {t for g in COGNITIVE_MIGS for t in MIGS[g] if t in (seed_tags or [])}
        for g, tags in MIGS.items():
            m = np.zeros(len(verb_profiles), dtype=bool)
            for t in tags:
                if t in candidate_anchor_masks:
                    m |= candidate_anchor_masks[t]
                    if t in seed_cog_tags: soul_match_mask |= candidate_anchor_masks[t]
            m_f, is_in_s, w = m.astype(np.float32), g in active_seed_migs, MIG_WEIGHTS.get(g, 1.0)
            if is_in_s:
                inter_w += m_f * w
                union_w += w
            elif g in STRUCTURAL_MIGS or g in SEMI_STRUCTURAL_MIGS: union_w += m_f * w
        identity_match = inter_w / (union_w + 1e-9)
        identity_match = np.maximum(identity_match, soul_match_mask.astype(np.float32) * 0.35)

    # 2. Vibe Signal (The Soul)
    # Re-integrate Topic Signal for consistency
    topic_sims = np.dot(topic_distributions.astype(np.float32), seed_topic_dist.astype(np.float32))
    topic_sims = np.clip(topic_sims, 0, 1)
    
    # Composite Vibe: Topics act as a multiplier, not a divisor
    vibe_sim = sem_sims * (1.0 + 0.5 * topic_sims)
    vibe_shield = (sem_sims > 0.35) | (topic_sims > 0.6)

    # 3. Mechanical Core (Linear Jaccard)
    v_c, v_s = verb_profiles.astype(np.float32), seed_verb_profile.astype(np.float32)
    tag_sims = np.sum(np.minimum(v_c, v_s), axis=1) / (np.sum(np.maximum(v_c, v_s), axis=1) + 1e-9)
    
    # 4. Oracle Blend
    id_power = np.where(vibe_shield | soul_match_mask | (identity_match > 0.8) | structural_seed_match, 1.0, 2.0)
    kernel = (tag_sims * (identity_match ** id_power)) * vibe_sim * tone_sim * diff_sim
    
    # 5. Hard Gates (Softer Floor)
    effective_floor = np.where(structural_seed_match | (identity_match > 0.8), 0.0, 0.05)
    kernel = np.where((sem_sims < effective_floor) & ~vibe_shield, 0.001, kernel) 
    
    if candidate_anchor_masks:
        kernel = apply_kernel_vetoes(kernel, candidate_masks=candidate_anchor_masks, seed_tags=seed_tags, seed_migs=active_seed_migs, is_2d=False, identity_match=identity_match, vibe_shield=vibe_shield, soul_match_mask=soul_match_mask)
        kernel = apply_identity_protection(kernel, candidate_masks=candidate_anchor_masks, seed_tags=seed_tags, seed_migs=active_seed_migs, is_2d=False, identity_match=identity_match, vibe_shield=vibe_shield)
        kernel = apply_thematic_clash(kernel, candidate_masks=candidate_anchor_masks, seed_migs=active_seed_migs, is_2d=False, identity_match=identity_match, vibe_shield=vibe_shield)
    
    if precalculated_masks and "title_hijack" in precalculated_masks:
        kernel *= np.where(precalculated_masks["title_hijack"] & (identity_match < 0.3) & ~vibe_shield, 0.0001, 1.0)

    final_kernel = np.maximum(kernel, 0.0)
    if return_components: return final_kernel, {'identity': identity_match, 'mechanical': tag_sims, 'vibe': vibe_sim, 'theme': sem_sims, 'tone': tone_sim, 'difficulty': diff_sim}
    return final_kernel

def to_z(x, ignore_zeros=False, clamp=None):
    x_array = np.asarray(x)
    if ignore_zeros:
        subset = x_array[np.abs(x_array) > 1e-5]
        if len(subset) == 0: subset = x_array
        mean, std = np.mean(subset, dtype=np.float64), np.std(subset, dtype=np.float64)
    else: mean, std = np.mean(x_array, dtype=np.float64), np.std(x_array, dtype=np.float64)
    z = (x_array - mean) / (std if std > EPSILON else 1.0)
    if clamp is not None: z = np.clip(z, clamp[0], clamp[1])
    return z.astype(np.float32)

def calculate_linear_scores(z_quality, z_date, z_pop, z_playtime, z_difficulty, z_price, tag_vectors, tag_norms, beta_tag, weights, tag_scaling_factor, dot_product_lambda, z_semantic=None, w_semantic=0.0, z_topic=None, w_topic=0.0, z_clamp_min=-3.0, z_clamp_max=3.0, dna_scaling_factor=1.0, intercept=5.0, tag_sim=None, x_kernel=None, x_graph=None, mig_features=None, seed_tag_sim=None, seed_sem_sim=None, seed_topic_sim=None, prompt_tag_sim=None, prompt_sem_sim=None, prompt_topic_sim=None):
    q, d, p, l, diff, pr = [np.clip(x, z_clamp_min, z_clamp_max) for x in [z_quality, z_date, z_pop, z_playtime, z_difficulty, z_price]]
    if tag_sim is None:
        beta_tag_arr = np.asarray(beta_tag, dtype=np.float32)
        if beta_tag_arr.size > 0:
            dot_products = np.dot(tag_vectors.astype(np.float32), beta_tag_arr)
            tag_sim = (dot_products / (tag_norms.astype(np.float32).reshape(-1) + dot_product_lambda)) * tag_scaling_factor
        else: tag_sim = np.zeros(len(q), dtype=np.float32)
    dna_tag_contrib, dna_sem_contrib, dna_top_contrib = (np.nan_to_num(tag_sim) * weights.get('tag_match', 0.0)), np.nan_to_num(z_semantic * w_semantic) if z_semantic is not None else 0.0, np.nan_to_num(z_topic * w_topic) if z_topic is not None else 0.0
    kernel_contrib = (np.nan_to_num(x_kernel) * weights.get('kernel_match', 0.0)) if x_kernel is not None else 0.0
    graph_contrib = (np.nan_to_num(x_graph) * weights.get('graph_match', 0.0)) if x_graph is not None else 0.0
    mig_contrib = 0.0
    if mig_features is not None:
        for group in MIGS.keys():
            feat_key = f"MIG_{group}"
            if feat_key in weights: mig_contrib += mig_features[group] * weights[feat_key]
    seed_contrib, prompt_contrib = 0.0, 0.0
    if seed_tag_sim is not None:
        signals = []
        if weights.get('tag_match', 0.0) > 1e-9: signals.append(seed_tag_sim)
        if w_semantic > 1e-9 and seed_sem_sim is not None: signals.append(seed_sem_sim)
        if w_topic > 1e-9 and seed_topic_sim is not None: signals.append(seed_topic_sim)
        if signals:
            consensus_sim = softmin_blend(signals, temperature=0.05)
            active_weights = [weights.get('tag_match', 1.0)]
            if w_semantic > 1e-9: active_weights.append(w_semantic)
            if w_topic > 1e-9: active_weights.append(w_topic)
            seed_contrib = consensus_sim * np.mean(active_weights)
    if prompt_sem_sim is not None:
        signals = []
        if weights.get('tag_match', 0.0) > 1e-9 and prompt_tag_sim is not None: signals.append(prompt_tag_sim)
        if w_semantic > 1e-9: signals.append(prompt_sem_sim)
        if w_topic > 1e-9: active_weights.append(w_topic)
        if signals:
            consensus_sim = softmin_blend(signals, temperature=0.05)
            active_weights = [w_semantic]
            if weights.get('tag_match', 0.0) > 1e-9: active_weights.append(weights.get('tag_match', 1.0))
            if w_topic > 1e-9: active_weights.append(w_topic)
            prompt_contrib = consensus_sim * np.mean(active_weights)
    scores = (np.nan_to_num(q) * weights.get('quality', 0.0) + np.nan_to_num(d) * weights.get('age', 0.0) + np.nan_to_num(p) * weights.get('popularity', 0.0) + np.nan_to_num(l) * weights.get('length', 0.0) + np.nan_to_num(diff) * weights.get('difficulty', 0.0) + np.nan_to_num(pr) * weights.get('price', 0.0) + dna_tag_contrib + dna_sem_contrib + dna_top_contrib + kernel_contrib + mig_contrib + seed_contrib + prompt_contrib)
    return (scores / dna_scaling_factor) + intercept

def calculate_hybrid_score(z_semantic, w_semantic, z_tag, w_tag, z_spps, w_spps, z_date, w_date, z_pop, w_pop, z_length, w_length, z_difficulty, w_difficulty, z_price, w_price):
    return ((z_semantic * w_semantic) + (z_tag * w_tag) + (z_spps * w_spps) + (z_date * w_date) + (z_pop * w_pop) + (z_length * w_length) + (z_difficulty * w_difficulty) + (z_price * w_price) + 5.0)

def normalize_string(s):
    import unicodedata
    if not s or not isinstance(s, str): return ""
    s = "".join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn').lower()
    s = re.sub(r'[™®©:]', '', s)
    s = re.sub(r'[^a-z0-9]', ' ', s)
    return " ".join(s.split())

def clean_release_date(date_str):
    if pd.isna(date_str) or date_str == "": return pd.NaT
    s, placeholders = str(date_str).strip(), {'coming soon', 'to be announced', 'maybe', 'tbd'}
    if s.lower() in placeholders: return pd.Timestamp.now().normalize() + pd.DateOffset(years=1)
    if re.search(r'\b(9998|6969|9000|2099)\b', s) or re.match(r'^[Qq][1-4]\s+\d{4}$', s): return pd.Timestamp.now().normalize() + pd.DateOffset(years=1)
    if re.match(r'^\d{4}-\d{2}-\d{2}$', s): return pd.to_datetime(s, errors='coerce')
    if re.match(r'^\d{4}$', s): return pd.to_datetime(s + "-07-01", errors='coerce')
    m1, m2 = re.match(r'^([A-Za-z]+)\s+(\d{4})$', s), re.match(r'^(\d{4})\s+([A-Za-z]+)$', s)
    if m1: return pd.to_datetime(f"{m1.group(1)} 15, {m1.group(2)}", errors='coerce')
    if m2: return pd.to_datetime(f"{m2.group(2)} 15, {m2.group(1)}", errors='coerce')
    return pd.to_datetime(s, errors='coerce')

def ensure_python_types(obj):
    if isinstance(obj, dict): return {k: ensure_python_types(v) for k, v in obj.items()}
    elif isinstance(obj, list): return [ensure_python_types(v) for v in obj]
    elif isinstance(obj, (np.int64, np.int32, np.int16, np.int8)): return int(obj)
    elif isinstance(obj, (np.float64, np.float32, np.float16)): return float(obj)
    elif isinstance(obj, np.ndarray): return ensure_python_types(obj.tolist())
    elif isinstance(obj, np.bool_): return bool(obj)
    return obj

def calculate_personalized_quality(q_global, p_plus_playtime):
    from scipy.stats import norm
    q_64 = q_global.astype(np.float64)
    pdf_q, cdf_q, sf_q = norm.pdf(q_64), norm.cdf(q_64), norm.sf(q_64)
    eps = 1e-12
    shift = (p_plus_playtime / (cdf_q + eps)) - ((1.0 - p_plus_playtime) / (sf_q + eps))
    return (q_64 + pdf_q * shift).astype(np.float32)

def calculate_dot_product_lambda(vectors):
    from scipy.stats import chi
    from common.constants import CHI_FIT_NORM_THRESHOLD, CHI_FIT_PERCENTILE
    lengths = np.linalg.norm(vectors, axis=1)
    subset_lengths = lengths[(lengths > 1e-6) & (lengths <= CHI_FIT_NORM_THRESHOLD)]
    if len(subset_lengths) > 10:
        df, loc, scale = chi.fit(subset_lengths)
        return chi.ppf(CHI_FIT_PERCENTILE, df, loc, scale)
    return 1.0

def safe_save_npy(path, data):
    import time
    if not path.endswith('.npy'): path += '.npy'
    if os.path.dirname(path): os.makedirs(os.path.dirname(path), exist_ok=True)
    temp_path = path + f".{os.getpid()}.tmp"
    np.save(temp_path, data)
    for attempt in range(10):
        try:
            if os.path.exists(path): os.replace(temp_path + ".npy", path)
            else: os.rename(temp_path + ".npy", path)
            return
        except OSError:
            garbage_path = path + f".old.{int(time.time())}.{attempt}"
            try:
                os.rename(path, garbage_path)
                os.rename(temp_path + ".npy", path)
                try: os.remove(garbage_path)
                except: pass
                return
            except OSError:
                if attempt < 9: time.sleep(1.0)
                else: raise

def softmin_blend(signals: list, temperature: float = SOFTMIN_TEMPERATURE):
    if not signals: return 0.0
    if len(signals) == 1: return signals[0]
    stack = np.stack(signals, axis=0)
    scaled = -stack / temperature
    max_val = np.max(scaled, axis=0)
    exp_vals = np.exp(scaled - max_val)
    weights = exp_vals / np.sum(exp_vals, axis=0)
    return np.sum(stack * weights, axis=0)

def get_base_filter_mask(metadata, english_only=False, remove_vr=False, remove_utilities=False, remove_delisted=False, remove_hollow=False, remove_unreleased=False):
    mask = np.ones(len(metadata), dtype=bool)
    if english_only: mask &= metadata['is_english'].values.astype(bool)
    if remove_vr: mask &= ~metadata['is_vr_only'].values.astype(bool)
    if remove_utilities: mask &= ~metadata['is_utility'].values.astype(bool)
    if remove_delisted: mask &= ~metadata['is_delisted'].values.astype(bool)
    if remove_hollow: mask &= ~metadata['is_hollow'].values.astype(bool)
    if remove_unreleased:
        from common.constants import METADATA_FILE
        if os.path.exists(METADATA_FILE):
            build_time = pd.Timestamp(os.path.getmtime(METADATA_FILE), unit='s')
            placeholders = ['coming soon', 'to be announced', 'maybe', 'tbd']
            is_placeholder = metadata['release_date'].fillna('').astype(str).str.lower().str.contains('|'.join(placeholders), regex=True).values
            is_future = (metadata['parsed_date'] > build_time).fillna(False).values | is_placeholder
            mask &= ~is_future.astype(bool)
    return mask

def extract_seed_metadata(indices, metadata):
    import ast
    seed_tags_soul_list, seed_tags_strict_list, seed_migs_list, seed_mature_flags = [], [], [], []
    all_soul_tags = set()
    for idx in indices:
        row = metadata.iloc[idx]
        tags_str = row['tags']
        try: tags_dict = ast.literal_eval(tags_str) if isinstance(tags_str, str) and tags_str.startswith('{') else {}
        except: tags_dict = {}
        max_v = max(tags_dict.values()) if tags_dict else 1.0
        s_tags_soul = {t for t, v in tags_dict.items() if v / max_v > 0.15}
        s_tags_strict = {t for t, v in tags_dict.items() if v / max_v > 0.35}
        seed_tags_soul_list.append(s_tags_soul)
        seed_tags_strict_list.append(s_tags_strict)
        all_soul_tags.update(s_tags_soul)
        s_migs = {group for group, tags in MIGS.items() if any(t in s_tags_soul for t in tags)}
        seed_migs_list.append(s_migs)
        seed_mature_flags.append(bool(row.get('mature_content', 0) > 0))
    active_narrative = [t for t in NARRATIVE_TAGS if t in all_soul_tags]
    is_cinematic = "Cinematic" in all_soul_tags
    return {'soul_tags_list': seed_tags_soul_list, 'strict_tags_list': seed_tags_strict_list, 'migs_list': seed_migs_list, 'mature_flags': seed_mature_flags, 'all_soul_tags': all_soul_tags, 'active_narrative': active_narrative, 'is_cinematic': is_cinematic}

def calculate_title_hijack_mask(seed_names, metadata):
    mask = np.zeros(len(metadata), dtype=bool)
    if not seed_names: return mask
    suffixes = {'remake', 'definitive edition', 'enhanced', 'special edition', 'remaster', 'bundle', 'goty', 'gold edition'}
    for seed_name in seed_names:
        clean_seed = normalize_string(seed_name)
        keywords = [k for k in re.split(r'[^a-zA-Z0-9]', str(seed_name).lower()) if len(k) > 3]
        if keywords:
            pattern = '|'.join(map(re.escape, keywords))
            title_match = metadata['name'].str.lower().str.contains(pattern, regex=True).values
            is_suffix = metadata['name'].str.lower().str.contains('|'.join(suffixes), regex=True).values
            mask |= (title_match & ~is_suffix)
    return mask

def calculate_jackalope_kernel_2d(verb_profiles, seed_verb_profiles, sem_vectors, sem_norms, seed_sem_vecs, seed_sem_norms, topic_distributions, seed_topic_dists, topic_means, topic_stds, candidate_mig_masks, seed_mig_masks, difficulty_z, seed_difficulty_z, tone_z, seed_tone_z, seed_tags=None, seed_migs=None, mature_content_flags=None, seed_mature_content_flags=None, graph_embeddings=None, seed_graph_vecs=None):
    import pandas as pd
    N_cand, M_seed = len(verb_profiles), len(seed_verb_profiles)
    
    # 1. Mechanical Signal (The Body)
    diff_sim = np.exp(-0.5 * ((difficulty_z[:, None].astype(np.float32) - seed_difficulty_z[None, :].astype(np.float32)) / 3.0)**2)
    tone_sim = np.exp(-0.5 * ((tone_z[:, None].astype(np.float32) - seed_tone_z[None, :].astype(np.float32)) / 3.0)**2)
    C_mig, S_mig = candidate_mig_masks.astype(np.float32), seed_mig_masks.astype(np.float32)
    w_vec = np.array([MIG_WEIGHTS.get(g, 1.0) for g in MIGS.keys()], dtype=np.float32)
    inter_w = np.dot(C_mig * w_vec, S_mig.T)
    c_sums_w, s_sums_w = np.dot(C_mig, w_vec), np.dot(S_mig, w_vec)
    union_w = c_sums_w[:, None] + s_sums_w[None, :] - inter_w
    identity_match = inter_w / (union_w + 1e-9)
    v_c, v_s = verb_profiles.astype(np.float32), seed_verb_profiles.astype(np.float32)
    tag_sims = np.zeros((N_cand, M_seed), dtype=np.float32)
    for i in range(M_seed):
        tag_sims[:, i] = np.sum(np.minimum(v_c, v_s[i]), axis=1) / (np.sum(np.maximum(v_c, v_s[i]), axis=1) + 1e-9)
    
    # 2. Thematic Signal (The Soul)
    sem_dots = np.dot(sem_vectors.astype(np.float32), seed_sem_vecs.astype(np.float32).T)
    from common.constants import SEMANTIC_DOT_PRODUCT_LAMBDA, TOPIC_GLOBAL_SCALING_FACTOR
    sem_sims = (sem_dots / (sem_norms[:, None].astype(np.float32) + SEMANTIC_DOT_PRODUCT_LAMBDA)) / (seed_sem_norms[None, :].astype(np.float32) + SEMANTIC_DOT_PRODUCT_LAMBDA)

    # Re-integrate Topic Signal
    topic_sims = np.dot(topic_distributions.astype(np.float32), seed_topic_dists.astype(np.float32).T)
    topic_sims = np.clip(topic_sims, 0, 1)

    # Composite Vibe: Topics act as a multiplier, not a divisor
    vibe_sim = sem_sims * (1.0 + 0.5 * topic_sims)
    vibe_shield = (sem_sims > 0.35) | (topic_sims > 0.6)

    # 3. Oracle Blend
    id_power = np.where(vibe_shield, 1.0, 2.0).astype(np.float32)
    kernel = (tag_sims * (identity_match ** id_power)) * vibe_sim * tone_sim * diff_sim

    # 4. Hard Gates (Softer Floor)
    kernel = np.where((sem_sims < 0.05) & ~vibe_shield, 0.001, kernel)
    return np.maximum(kernel.astype(np.float32), 0.0)

