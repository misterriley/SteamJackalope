import numpy as np
import re
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
    "PUZZLE_LOGIC": {"Logic", "Sokoban", "Programming", "Coding", "Automation", "Word Game"},
    "PUZZLE_SPATIAL": {"Puzzle", "Puzzle Platformer", "3D Platformer", "Hidden Object"},
    "PLATFORMER": {"Precision Platformer", "Runner", "Platformer"},
    "STRATEGY": {"Turn-Based Strategy", "Turn-Based Tactics", "RTS", "Real-Time", "Action RTS", "Tower Defense", "4X"},
    "MANAGEMENT": {"Colony Sim", "City Builder", "Management", "Shop Keeper", "Economy", "Resource Management", "Inventory Management"},
    "BUILDING": {"Building", "Base Building", "Sandbox"},
    "SURVIVAL": {"Survival", "Survival Horror", "Open World Survival Craft"},
    "ROGUELIKE": {"Roguelike", "Roguelite", "Action Roguelike", "Traditional Roguelike", "Roguelike Deckbuilder"},
    "NARRATIVE_VN": {"Visual Novel", "Anime", "Otome"},
    "DATING_SIM": {"Dating Sim"},
    "NARRATIVE_STORY": {"Story Rich", "Choices Matter", "Multiple Endings", "Cinematic", "Emotional", "Interactive Fiction", "Romance"},
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
    "SCI_FI": {"Sci-fi", "Futuristic", "Cyberpunk", "Space", "Space Sim"},
    "FANTASY": {"Fantasy", "Magic", "Medieval", "Dark Fantasy"},
    "HISTORICAL": {"Historical", "World War II", "World War I", "History"},
    "SURREAL": {"Surreal", "Psychedelic", "Stylized", "Experimental"}
}

def to_z(x, ignore_zeros=False):
    """
    Calculates the z-score of a numerical array.
    """
    x_array = np.asarray(x)
    if ignore_zeros:
        subset = x_array[np.abs(x_array) > 1e-5]
        if len(subset) == 0:
            subset = x_array
        mean = np.mean(subset, dtype=np.float64)
        std = np.std(subset, dtype=np.float64)
    else:
        mean = np.mean(x_array, dtype=np.float64)
        std = np.std(x_array, dtype=np.float64)
    z = (x_array - mean) / (std if std > EPSILON else 1.0)
    return z

def calculate_linear_scores(
    z_quality, z_date, z_pop, z_playtime, z_difficulty, z_price,
    tag_vectors, tag_norms, beta_tag,
    weights,
    tag_scaling_factor,
    dot_product_lambda,
    z_semantic=None,
    w_semantic=0.0,
    z_topic=None,
    w_topic=0.0,
    z_clamp_min=-3.0,
    z_clamp_max=3.0,
    dna_scaling_factor=1.0,
    intercept=5.0,
    tag_sim=None,
    # New Consensus Inputs
    seed_tag_sim=None,
    seed_sem_sim=None,
    seed_topic_sim=None,
    prompt_tag_sim=None,
    prompt_sem_sim=None,
    prompt_topic_sim=None
):
    """
    Unified linear scoring function for Taste DNA parity with Consensus Support.
    """
    # 1. Apply Clamping to Metadata
    q = np.clip(z_quality, z_clamp_min, z_clamp_max)
    d = np.clip(z_date, z_clamp_min, z_clamp_max)
    p = np.clip(z_pop, z_clamp_min, z_clamp_max)
    l = np.clip(z_playtime, z_clamp_min, z_clamp_max)
    diff = np.clip(z_difficulty, z_clamp_min, z_clamp_max)
    pr = np.clip(z_price, z_clamp_min, z_clamp_max)
    
    # 2. DNA Components (Linear Aggregate)
    if tag_sim is None:
        beta_tag_arr = np.asarray(beta_tag, dtype=np.float32)
        if beta_tag_arr.size > 0:
            dot_products = np.dot(tag_vectors.astype(np.float32), beta_tag_arr)
            denom = tag_norms.astype(np.float32).reshape(-1) + dot_product_lambda
            tag_sim = (dot_products / denom) * tag_scaling_factor
        else:
            tag_sim = np.zeros(len(q), dtype=np.float32)
    
    dna_tag_contrib = (np.nan_to_num(tag_sim) * weights.get('tag_match', 0.0))
    dna_sem_contrib = np.nan_to_num(z_semantic * w_semantic) if z_semantic is not None else 0.0
    dna_top_contrib = np.nan_to_num(z_topic * w_topic) if z_topic is not None else 0.0

    # 3. Consensus Components (Exemplar Seeds & Prompts)
    seed_contrib = 0.0
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

    prompt_contrib = 0.0
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

    # 4. Final Summation
    scores = (
        np.nan_to_num(q) * weights.get('quality', 0.0) +
        np.nan_to_num(d) * weights.get('age', 0.0) +
        np.nan_to_num(p) * weights.get('popularity', 0.0) +
        np.nan_to_num(l) * weights.get('length', 0.0) +
        np.nan_to_num(diff) * weights.get('difficulty', 0.0) +
        np.nan_to_num(pr) * weights.get('price', 0.0) +
        dna_tag_contrib +
        dna_sem_contrib +
        dna_top_contrib +
        seed_contrib +
        prompt_contrib
    )
    
    return (scores / dna_scaling_factor) + intercept

def calculate_hybrid_score(
    z_semantic, w_semantic,
    z_tag, w_tag,
    z_spps, w_spps,
    z_date, w_date,
    z_pop, w_pop,
    z_length, w_length,
    z_difficulty, w_difficulty,
    z_price, w_price
):
    """
    Calculates the final hybrid score for recommender mode (Manual Sliders).
    """
    return (
        (z_semantic * w_semantic) +
        (z_tag * w_tag) +
        (z_spps * w_spps) +
        (z_date * w_date) +
        (z_pop * w_pop) +
        (z_length * w_length) +
        (z_difficulty * w_difficulty) +
        (z_price * w_price) +
        5.0 # Neutral Anchor
    )

def ensure_python_types(obj):
    """
    Recursively converts NumPy types to standard Python types for JSON serialization.
    """
    import pandas as pd
    if isinstance(obj, dict):
        return {k: ensure_python_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [ensure_python_types(v) for v in obj]
    elif isinstance(obj, (np.int64, np.int32, np.int16, np.int8)):
        return int(obj)
    elif isinstance(obj, (np.float64, np.float32, np.float16)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return ensure_python_types(obj.tolist())
    elif isinstance(obj, np.bool_):
        return bool(obj)
    return obj

def calculate_personalized_quality(q_global, p_plus_playtime):
    """
    Adjusts global quality scores (probits) to reflect the expected experience
    at a specific playtime.
    """
    from scipy.stats import norm
    q_64 = q_global.astype(np.float64)
    pdf_q = norm.pdf(q_64)
    cdf_q = norm.cdf(q_64)
    sf_q = norm.sf(q_64)
    eps = 1e-12
    shift = (p_plus_playtime / (cdf_q + eps)) - ((1.0 - p_plus_playtime) / (sf_q + eps))
    personalized_q = q_64 + pdf_q * shift
    return personalized_q.astype(np.float32)

def calculate_dot_product_lambda(vectors):
    """
    Calculates DOT_PRODUCT_LAMBDA via Chi-distribution fit.
    """
    from scipy.stats import chi
    from common.constants import CHI_FIT_NORM_THRESHOLD, CHI_FIT_PERCENTILE
    lengths = np.linalg.norm(vectors, axis=1)
    subset_mask = (lengths > 1e-6) & (lengths <= CHI_FIT_NORM_THRESHOLD)
    subset_lengths = lengths[subset_mask]
    if len(subset_lengths) > 10:
        df, loc, scale = chi.fit(subset_lengths)
        data_driven_lambda = chi.ppf(CHI_FIT_PERCENTILE, df, loc, scale)
    else:
        data_driven_lambda = 1.0
    return data_driven_lambda

def safe_save_npy(path, data):
    """
    Saves a NumPy array to a file safely handling Windows file locking.
    """
    import os
    import time
    if not path.endswith('.npy'):
        path += '.npy'
    dir_name = os.path.dirname(path)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)
    temp_path = path + f".{os.getpid()}.tmp"
    actual_temp_file = temp_path + ".npy"
    np.save(temp_path, data)
    max_retries = 10
    retry_delay = 1.0
    for attempt in range(max_retries):
        try:
            if os.path.exists(path):
                os.replace(actual_temp_file, path)
            else:
                os.rename(actual_temp_file, path)
            return
        except OSError:
            garbage_path = path + f".old.{int(time.time())}.{attempt}"
            try:
                os.rename(path, garbage_path)
                os.rename(actual_temp_file, path)
                try:
                    os.remove(garbage_path)
                except:
                    pass
                return
            except OSError:
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                else:
                    raise

def softmin_blend(signals: list, temperature: float = SOFTMIN_TEMPERATURE):
    """
    Blends multiple similarity signals using a softmin-weighted average.
    """
    if not signals:
        return 0.0
    if len(signals) == 1:
        return signals[0]
        
    stack = np.stack(signals, axis=0) # (num_signals, num_games)
    scaled = -stack / temperature
    max_val = np.max(scaled, axis=0)
    exp_vals = np.exp(scaled - max_val)
    weights = exp_vals / np.sum(exp_vals, axis=0)
    return np.sum(stack * weights, axis=0)

def soft_gate(signal, threshold, temperature=0.01, mode='high_pass'):
    """
    A differentiable sigmoid gate that transitions smoothly around a threshold.
    mode='high_pass': 1.0 above threshold, 0.0 below.
    mode='low_pass': 0.0 above threshold, 1.0 below.
    """
    val = (signal - threshold) / (temperature + 1e-9)
    if mode == 'high_pass':
        return 1.0 / (1.0 + np.exp(-np.clip(val, -50, 50)))
    else:
        return 1.0 / (1.0 + np.exp(np.clip(val, -50, 50)))

def calculate_jackalope_kernel(
    verb_profiles, seed_verb_profile,
    sem_vectors, sem_norms, seed_sem_vec, seed_sem_norm,
    topic_distributions, seed_topic_dist,
    topic_means, topic_stds,
    tag_scaling_factor, dot_product_lambda,
    sem_scaling_factor, sem_lambda,
    topic_scaling_factor=0.1,
    # Mature Content Flag (Steam Banner)
    mature_content_flags=None,
    seed_mature_content=False,
    # Veto/Rescue Metadata
    seed_migs=None,
    seed_tags=None,
    candidate_anchor_masks=None,
    active_narrative_seed=None,
    is_cinematic_seed=False,
    is_crpg_seed=False,
    # PRE-CALCULATED MASKS
    precalculated_masks=None,
    # Similarity Dimensions
    difficulty_z=None,
    seed_difficulty_z=None,
    tone_z=None,
    seed_tone_z=None,
    temperature=0.01,
    return_components=False
):
    """
    Jackalope Kernel v4.1: Advanced spirit-aware blender with identity-pollution protection.
    """
    import pandas as pd

    # 0. Spirit Dimensions (Gaussian) - Ultra High Sigma (3.0) for discovery
    diff_sim = np.exp(-0.5 * ((difficulty_z.astype(np.float32) - float(seed_difficulty_z or 0)) / 3.0)**2) if difficulty_z is not None else 1.0
    tone_sim = np.exp(-0.5 * ((tone_z.astype(np.float32) - float(seed_tone_z or 0)) / 3.0)**2) if tone_z is not None else 1.0

    # 1. Identity Overlap Signal (Jaccard on MIGs)
    active_seed_migs = set(seed_migs or [])
    identity_match = np.ones(len(verb_profiles), dtype=np.float32)
    
    if candidate_anchor_masks:
        intersection = np.zeros(len(verb_profiles), dtype=np.float32)
        union = np.zeros(len(verb_profiles), dtype=np.float32)
        
        for group, tags in MIGS.items():
            m = np.zeros(len(verb_profiles), dtype=bool)
            for t in tags:
                if t in candidate_anchor_masks: m |= candidate_anchor_masks[t]
            
            is_in_seed = group in active_seed_migs
            m_float = m.astype(np.float32)
            
            if is_in_seed:
                intersection += m_float
                union += 1.0
            else:
                union += m_float
                
        identity_match = intersection / (union + 1e-9)

    # 2. Mechanical Core (Verbs)
    ALPHA = 0.1
    intersection = np.sum(np.minimum(verb_profiles.astype(np.float32), seed_verb_profile.astype(np.float32)), axis=1)
    union = np.sum(np.maximum(verb_profiles.astype(np.float32), seed_verb_profile.astype(np.float32)), axis=1)
    tag_sims = intersection / (intersection + ALPHA * (union - intersection) + 1e-9)

    # 3. Vibe Similarity
    sem_sims_raw = (np.dot(sem_vectors.astype(np.float32), seed_sem_vec.astype(np.float32)) /
                    (sem_norms + sem_lambda))
    sem_sims = sem_sims_raw / (seed_sem_norm + sem_lambda)

    fz_unit_seed = (seed_topic_dist.astype(np.float32) - topic_means) / (topic_stds + 1e-9)
    fz_unit_seed[fz_unit_seed < 0.0] = 0
    fz_unit_seed /= (np.linalg.norm(fz_unit_seed) + 1e-9)

    topic_sims = np.zeros(len(verb_profiles), dtype=np.float32)
    batch_size = 50000
    for i in range(0, len(verb_profiles), batch_size):
        end = min(i + batch_size, len(verb_profiles))
        bz = (topic_distributions[i:end].astype(np.float32) - topic_means) / (topic_stds + 1e-9)
        bz[bz < 0.0] = 0
        bn = np.linalg.norm(bz, axis=1, keepdims=True) + 1e-9
        topic_sims[i:end] = np.dot(bz / bn, fz_unit_seed)

    sem_cdf = pd.Series(sem_sims).rank(pct=True).values.astype(np.float32)
    topic_cdf = pd.Series(topic_sims).rank(pct=True).values.astype(np.float32)
    vibe_sim = 0.5 * sem_cdf + 0.5 * topic_cdf

    # 4. Multi-Modal Blend
    # Kernel = Identity * Vibe * Spirit
    kernel = (tag_sims * (identity_match ** KERNEL_IDENTITY_POWER)) * vibe_sim * tone_sim * diff_sim

    # 5. VETO GATES
    if candidate_anchor_masks:
        veto_multiplier = np.ones(len(verb_profiles), dtype=np.float32)
        
        # A. Perspective Gate
        perspectives_2d = ["2D", "Side Scroller"]
        perspectives_3d = ["3D", "Third Person", "First-Person", "Isometric", "Third-Person Shooter", "FPS"]
        
        is_2d_seed = any(p in (seed_tags or []) for p in perspectives_2d)
        is_3d_seed = any(p in (seed_tags or []) for p in perspectives_3d)
        
        m_2d = np.zeros(len(kernel), dtype=bool)
        for p in perspectives_2d:
            if p in candidate_anchor_masks: m_2d |= candidate_anchor_masks[p]
            
        m_3d = np.zeros(len(kernel), dtype=bool)
        for p in perspectives_3d:
            if p in candidate_anchor_masks: m_3d |= candidate_anchor_masks[p]
            
        if is_2d_seed and not is_3d_seed:
            veto_multiplier *= (KERNEL_PERSPECTIVE_PENALTY + (1.0 - KERNEL_PERSPECTIVE_PENALTY) * m_2d.astype(float))
        elif is_3d_seed and not is_2d_seed:
            veto_multiplier *= (KERNEL_PERSPECTIVE_PENALTY + (1.0 - KERNEL_PERSPECTIVE_PENALTY) * m_3d.astype(float))

        # B. NSFW Gate (Multi-Tiered Strictness)
        strong_nsfw_tags = {"Hentai", "Nudity"}
        seed_has_strong_nsfw = any(t in strong_nsfw_tags for t in (seed_tags or []))
        
        if not seed_has_strong_nsfw:
            # Candidate has strong NSFW but seed doesn't -> Slap hard
            m_strong = np.zeros(len(kernel), dtype=bool)
            for t in strong_nsfw_tags:
                if t in candidate_anchor_masks: m_strong |= candidate_anchor_masks[t]
            veto_multiplier *= (KERNEL_STRONG_NSFW_PENALTY + (1.0 - KERNEL_STRONG_NSFW_PENALTY) * (~m_strong).astype(float))
            
            # If seed also lacks generic sexual content, slap candidates that HAVE it
            seed_is_nsfw = bool(seed_mature_content) or any(t in {"Mature", "Violent", "Gore", "Sexual Content"} for t in (seed_tags or []))
            if not seed_is_nsfw:
                nsfw_mask = np.zeros(len(kernel), dtype=bool)
                if mature_content_flags is not None: nsfw_mask |= mature_content_flags.astype(bool)
                for t in {"Sexual Content", "Mature"}:
                    if t in candidate_anchor_masks: nsfw_mask |= candidate_anchor_masks[t]
                veto_multiplier *= (KERNEL_MATURE_NSFW_PENALTY + (1.0 - KERNEL_MATURE_NSFW_PENALTY) * (~nsfw_mask).astype(float))

        # C. VR Gate
        seed_is_vr = "VR" in active_seed_migs or "VR Only" in (seed_tags or [])
        if not seed_is_vr:
            m_vr = candidate_anchor_masks.get("VR", np.zeros(len(kernel), dtype=bool)) | candidate_anchor_masks.get("VR Only", np.zeros(len(kernel), dtype=bool))
            veto_multiplier *= (KERNEL_VR_PENALTY + (1.0 - KERNEL_VR_PENALTY) * (~m_vr).astype(float))

        kernel *= veto_multiplier

    # 6. IDENTITY POLLUTION PROTECTION (Strict Group Vetoes)
    # If a seed lacks a noisy mechanical group that the candidate HAS, penalize.
    if active_seed_migs:
        # HARD VETO GROUPS - Game-breaking mechanical pivots
        HARD_POLLUTION = {"VEHICLE_SIM", "SIMULATION", "STRATEGY", "MANAGEMENT", "ROGUELIKE", "VR", "FMV", "POINT_AND_CLICK", "DATING_SIM"}
        # SOFT VETO GROUPS - Significant but compatible mechanical pivots
        SOFT_POLLUTION = {"METROIDVANIA", "NARRATIVE_VN", "SURVIVAL", "BUILDING", "SHOOTER", "MELEE_ACTION"}
        
        for group in HARD_POLLUTION:
            if group not in active_seed_migs:
                seed_has_soul_tag = any(t in (seed_tags or []) for t in MIGS[group])
                if not seed_has_soul_tag:
                    m = np.zeros(len(kernel), dtype=bool)
                    for t in MIGS[group]:
                        if t in candidate_anchor_masks: m |= candidate_anchor_masks[t]
                    kernel *= (KERNEL_HARD_POLLUTION_PENALTY + (1.0 - KERNEL_HARD_POLLUTION_PENALTY) * (~m).astype(float))
                    
        for group in SOFT_POLLUTION:
            if group not in active_seed_migs:
                seed_has_soul_tag = any(t in (seed_tags or []) for t in MIGS[group])
                if not seed_has_soul_tag:
                    m = np.zeros(len(kernel), dtype=bool)
                    for t in MIGS[group]:
                        if t in candidate_anchor_masks: m |= candidate_anchor_masks[t]
                    kernel *= (KERNEL_SOFT_POLLUTION_PENALTY + (1.0 - KERNEL_SOFT_POLLUTION_PENALTY) * (~m).astype(float))

    # 7. THEMATIC CLASH PROTECTION
    if active_seed_migs:
        # If seed is Sci-fi but candidate is Fantasy (or vice versa), slap it.
        seed_is_scifi = "SCI_FI" in active_seed_migs
        seed_is_fantasy = "FANTASY" in active_seed_migs
        
        if seed_is_scifi and not seed_is_fantasy:
            m_fantasy = np.zeros(len(kernel), dtype=bool)
            for t in MIGS["FANTASY"]:
                if t in candidate_anchor_masks: m_fantasy |= candidate_anchor_masks[t]
            kernel *= (KERNEL_THEMATIC_CLASH_PENALTY + (1.0 - KERNEL_THEMATIC_CLASH_PENALTY) * (~m_fantasy).astype(float))
            
        if seed_is_fantasy and not seed_is_scifi:
            m_scifi = np.zeros(len(kernel), dtype=bool)
            for t in MIGS["SCI_FI"]:
                if t in candidate_anchor_masks: m_scifi |= candidate_anchor_masks[t]
            kernel *= (KERNEL_THEMATIC_CLASH_PENALTY + (1.0 - KERNEL_THEMATIC_CLASH_PENALTY) * (~m_scifi).astype(float))

    # 8. SPECIFIC BOOSTS
    # Puzzle Resonance
    if any(g in active_seed_migs for g in ["PUZZLE_LOGIC", "PUZZLE_SPATIAL"]):
        p_mask = np.zeros(len(kernel), dtype=bool)
        for g in ["PUZZLE_LOGIC", "PUZZLE_SPATIAL"]:
            for t in MIGS[g]:
                if t in candidate_anchor_masks: p_mask |= candidate_anchor_masks[t]
        kernel += np.where(p_mask & (vibe_sim > 0.7), 0.1, 0.0)

    # 8. SEMANTIC RESCUE
    rescue_boost = np.where(sem_cdf > 0.999, KERNEL_RESCUE_THRESHOLD, 0.0)
    kernel += rescue_boost

    # 9. TITLE HIJACK PROTECTION
    if precalculated_masks and "title_hijack" in precalculated_masks:
        hijack_mask = precalculated_masks["title_hijack"]
        # If mechanical identity is low but name matches, it's likely a hijacker.
        kernel *= np.where(hijack_mask & (identity_match < 0.4), 0.01, 1.0)

    # 10. Consensus Floor
    kernel *= soft_gate(vibe_sim, threshold=0.001, temperature=0.001)
    
    final_kernel = np.maximum(kernel, 0.0)
    if return_components:
        return final_kernel, {'identity': identity_match, 'mechanical': tag_sims, 'vibe': vibe_sim, 'tone': tone_sim, 'difficulty': diff_sim}
    return final_kernel

def fast_jsd_similarity(p, Q_matrix, mean=None, std=None):
    """
    Calculates Jensen-Shannon Similarity (1 - JSD).
    """
    eps = 1e-12
    p = np.clip(p, eps, 1.0)
    Q = np.clip(Q_matrix, eps, 1.0)
    p = p / np.sum(p)
    Q = Q / np.sum(Q, axis=-1, keepdims=True)
    m = 0.5 * (p + Q)
    term1 = p * np.log(p / m)
    term2 = Q * np.log(Q / m)
    js_div = 0.5 * (np.sum(term1, axis=-1) + np.sum(term2, axis=-1))
    return 1.0 - np.sqrt(js_div)

def calculate_jackalope_kernel_2d(
    verb_profiles, seed_verb_profiles,
    sem_vectors, sem_norms, seed_sem_vecs, seed_sem_norms,
    topic_distributions, seed_topic_dists,
    topic_means, topic_stds,
    candidate_mig_masks,
    seed_mig_masks,
    difficulty_z,
    seed_difficulty_z,
    tone_z,
    seed_tone_z
):
    """
    Fully vectorized 2D Jackalope Kernel.
    Computes a (N_candidates, M_seeds) matrix in one pass.
    Used for bulk continuous feature estimation.
    """
    import pandas as pd
    N_cand = len(verb_profiles)
    M_seed = len(seed_verb_profiles)
    
    # 0. Spirit Dimensions
    diff_sim = np.exp(-0.5 * ((difficulty_z[:, None] - seed_difficulty_z[None, :]) / 3.0)**2)
    tone_sim = np.exp(-0.5 * ((tone_z[:, None] - seed_tone_z[None, :]) / 3.0)**2)
    
    # 1. Identity Overlap Signal (Jaccard on MIGs)
    C_mig = candidate_mig_masks.astype(np.float32)
    S_mig = seed_mig_masks.astype(np.float32)
    
    inter_count = np.dot(C_mig, S_mig.T)
    c_sums = np.sum(C_mig, axis=1)
    s_sums = np.sum(S_mig, axis=1)
    
    union_count = c_sums[:, None] + s_sums[None, :] - inter_count
    identity_match = inter_count / (union_count + 1e-9)
    
    # 2. Mechanical Core (Verbs)
    ALPHA = 0.1
    tag_sims = np.zeros((N_cand, M_seed), dtype=np.float32)
    for i in range(M_seed):
        inter = np.sum(np.minimum(verb_profiles, seed_verb_profiles[i]), axis=1)
        union = np.sum(np.maximum(verb_profiles, seed_verb_profiles[i]), axis=1)
        tag_sims[:, i] = inter / (inter + ALPHA * (union - inter) + 1e-9)
        
    # 3. Vibe Similarity
    from common.constants import SEMANTIC_DOT_PRODUCT_LAMBDA
    sem_dots = np.dot(sem_vectors, seed_sem_vecs.T)
    sem_sims_raw = sem_dots / (sem_norms[:, None] + SEMANTIC_DOT_PRODUCT_LAMBDA)
    sem_sims = sem_sims_raw / (seed_sem_norms[None, :] + SEMANTIC_DOT_PRODUCT_LAMBDA)
    
    bz = (topic_distributions - topic_means) / (topic_stds + 1e-9)
    bz = np.maximum(bz, 0.0)
    bn = np.linalg.norm(bz, axis=1, keepdims=True) + 1e-9
    bz_norm = bz / bn
    
    sz = (seed_topic_dists - topic_means) / (topic_stds + 1e-9)
    sz = np.maximum(sz, 0.0)
    sn = np.linalg.norm(sz, axis=1, keepdims=True) + 1e-9
    sz_norm = sz / sn
    
    topic_sims = np.dot(bz_norm, sz_norm.T)
    
    sem_cdf = pd.DataFrame(sem_sims).rank(pct=True, axis=0).values.astype(np.float32)
    topic_cdf = pd.DataFrame(topic_sims).rank(pct=True, axis=0).values.astype(np.float32)
    vibe_sim = 0.5 * sem_cdf + 0.5 * topic_cdf
    
    # 4. Multi-Modal Blend
    kernel = (tag_sims * (identity_match ** KERNEL_IDENTITY_POWER)) * vibe_sim * tone_sim * diff_sim

    # 5. SEMANTIC RESCUE
    rescue_boost = np.where(sem_cdf > 0.999, KERNEL_RESCUE_THRESHOLD, 0.0)
    kernel += rescue_boost

    # 6. Consensus Floor
    kernel *= soft_gate(vibe_sim, threshold=0.001, temperature=0.001)
    
    return np.maximum(kernel, 0.0)
