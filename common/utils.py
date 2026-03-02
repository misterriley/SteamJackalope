import numpy as np
import re
from common.constants import EPSILON, Z_SCORE_CLAMP_MIN, Z_SCORE_CLAMP_MAX, SOFTMIN_TEMPERATURE

# --- Mechanical Identity Groups (MIGs) ---
MIGS = {
    "SHOOTER": {"FPS", "Shooter", "Third-Person Shooter", "Arena Shooter", "Hero Shooter", "Looter Shooter", "Extraction Shooter", "Boomer Shooter", "On-Rails Shooter", "Immersive Sim", "Top-Down Shooter", "Twin Stick Shooter", "Battle Royale", "Tactical FPS"},
    "BULLET_HELL": {"Bullet Hell", "Shoot 'Em Up", "Twin Stick Shooter", "Top-Down Shooter"},
    "SPATIAL_PUZZLE": {"Puzzle", "First-Person", "Third Person", "3D", "Exploration", "Walking Simulator", "Puzzle Platformer"},
    "LOGIC_PUZZLE": {"Logic", "Sokoban", "Abstract", "Minimalist", "Programming", "Coding", "Automation"},
    "CASUAL_PUZZLE": {"Hidden Object", "Point & Click", "Match 3", "Trivia", "Word Game", "Board Game"},
    "COMPETITIVE": {"PvP", "eSports", "Competitive", "Battle Royale", "Multiplayer"},
    "STEALTH": {"Stealth", "Assassin", "Immersive Sim"},
    "WALKING_SIM": {"Walking Simulator", "Interactive Fiction", "Cinematic", "Story Rich", "Choices Matter", "Multiple Endings"},
    "IDLE_CLICKER": {"Idler", "Clicker", "Incremental", "Idle"},
    "SINGLEPLAYER": {"Singleplayer"},
    "FAMILY_FRIENDLY": {"Family Friendly"},
    "PLATFORMER": {"Precision Platformer", "2D Platformer", "3D Platformer", "Platformer", "Runner", "Puzzle Platformer", "Sokoban"},
    "FIGHTING": {"Fighting", "2D Fighter", "3D Fighter", "Boxing", "Wrestling", "Beat 'em up"},
    "MELEE_ACTION": {"Souls-like", "Spectacle fighter", "Hack and Slash", "Beat 'em up", "Character Action Game", "Musou", "Swordplay", "Action Roguelike", "Action RPG"},
    "DECKBUILDER": {"Card Battler", "Roguelike Deckbuilder", "Trading Card Game", "Card Game", "Deckbuilding", "Board Game"},
    "TACTICAL": {"Tactical", "Real Time Tactics", "Turn-Based Tactics", "Turn-Based Strategy", "Tactical RPG", "Strategy RPG", "Wargame", "Tactical FPS"},
    "GRAND_STRATEGY": {"Grand Strategy", "4X", "Wargame"},
    "STRATEGY_RT": {"RTS", "Real-Time", "Action RTS", "Tower Defense"},
    "SURVIVAL": {"Survival", "Survival Horror", "Open World Survival Craft", "Inventory Management", "Resource Management"},
    "MANAGEMENT": {"Management", "Colony Sim", "City Builder", "Resource Management", "Time Management", "Inventory Management", "Shop Keeper", "Base Building", "Automation", "Farming Sim", "Life Sim", "Economy"},
    "BUILDING": {"Building", "Base Building", "Sandbox"},
    "AUTOMATION": {"Automation", "Programming", "Coding"},
    "EDUCATION": {"Education", "Science", "Math", "Typing", "Spelling"},
    "CRPG": {"CRPG", "Party-Based RPG", "Tactical RPG", "Strategy RPG", "Dungeon Crawler", "Hack and Slash", "JRPG", "Looter Shooter", "Action RPG", "Immersive Sim"},
    "POINT_AND_CLICK": {"Point & Click", "Hidden Object", "Visual Novel", "Interactive Fiction", "Dating Sim", "Otome"},
    "LIFE_SIM": {"Life Sim", "Farming Sim", "Social Beam", "Dating Sim", "Otome"},
    "METROIDVANIA": {"Metroidvania", "Roguevania"},
    "ROGUELIKE": {"Roguelike", "Roguelite", "Action Roguelike", "Traditional Roguelike", "Roguevania", "Roguelike Deckbuilder", "Rogue-like", "Rogue-lite", "Dungeon Crawler"},
    "BOARD_GAME": {"Board Game", "Trivia", "Chess", "Tabletop", "Solitaire", "Word Game"},
    "RACING": {"Racing", "Driving", "Automobile Sim", "Combat Racing", "Vehicular Combat"},
    "FLIGHT_SPACE": {"Flight", "Space Sim", "Space", "Sci-fi", "Sailing"},
    "RHYTHM": {"Rhythm", "Music"},
    "MOBA": {"MOBA", "Hero Shooter"},
    "HORROR": {"Horror", "Survival Horror", "Psychological Horror"},
    "TURN_BASED": {"Turn-Based", "Turn-Based Strategy", "Turn-Based Combat", "Turn-Based Tactics", "JRPG", "Turn-Based RPG", "4X", "Board Game"},
    "SPORTS": {"Sports", "Football", "Soccer", "Basketball", "Golf", "Skating", "Extreme Sports"},
    "SOULSLIKE": {"Souls-like", "Difficult", "Action RPG"},
    "VR": {"VR", "VR Only"}
}

NARRATIVE_TAGS = {"Visual Novel", "Interactive Fiction", "Story Rich", "Multiple Endings", "Choices Matter", "Narrative", "Character Customization", "Lore-Rich", "Emotional", "Cinematic"}
HORROR_MARKERS = {"Horror", "Survival Horror", "Psychological Horror", "Gore", "Violent"}
HARD_ANCHORS = {"Platformer", "Puzzle", "Strategy", "RPG", "Roguelike", "Souls-like", "Metroidvania", "JRPG", "Survival", "Visual Novel", "FPS", "First-Person", "Third Person", "Third-Person Shooter", "Shooter", "Walking Simulator"}

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
        if w_topic > 1e-9 and prompt_topic_sim is not None: signals.append(prompt_topic_sim)
        
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
    # Sigmoid function: 1 / (1 + exp(-x))
    # We scale by temperature to control the sharpness of the transition
    # Added clipping to prevent overflow in exp
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
    mature_content_flags=None, # np.ndarray of booleans for candidates
    seed_mature_content=False, # boolean flag for seed
    # Veto/Rescue Metadata
    seed_migs=None, # Set of MIG group names (e.g. {"SHOOTER", "ROGUELIKE"})
    seed_tags=None, # Set of specific mechanical tags for hard vetoes (e.g. {"Platformer"})
    candidate_anchor_masks=None, # dict: tag_name -> boolean mask
    active_narrative_seed=None,
    is_cinematic_seed=False,
    is_crpg_seed=False,
    # PRE-CALCULATED MASKS (Optimization)
    precalculated_masks=None, # dict: key -> boolean mask
    # Similarity Dimensions
    difficulty_z=None, # np.ndarray of Z-scores for candidates
    seed_difficulty_z=None, # float Z-score for seed
    tone_z=None, # np.ndarray of Z-scores for candidates
    seed_tone_z=None, # float Z-score for seed
    # Temperature (Control discovery vs discipline)
    temperature=0.01,
    return_components=False
):
    """
    The Jackalope Kernel: A multi-modal 'Mechanical Identity' measure.
    Combines Verb Profiles (Soft Jaccard), Semantics, and Topics with structural soft-vetoes.
    """
    import pandas as pd

    # 0. Similarity Dimensions (Gaussian)
    diff_sim = np.ones(len(verb_profiles), dtype=np.float32)
    if difficulty_z is not None and seed_difficulty_z is not None:
        diff_sim = np.exp(-0.5 * (difficulty_z.astype(np.float32) - float(seed_difficulty_z))**2)
        
    tone_sim = np.ones(len(verb_profiles), dtype=np.float32)
    if tone_z is not None and seed_tone_z is not None:
        # Use Sigma=0.7 for tone to be more selective about "Soul" matches
        tone_sim = np.exp(-0.5 * ((tone_z.astype(np.float32) - float(seed_tone_z)) / 0.7)**2)

    # 1. Mechanical Identity Groups (MIGs) - First Pass Skill Barrier
    skill_multiplier = np.ones(len(verb_profiles), dtype=np.float32)
    skill_jaccard = np.ones(len(verb_profiles), dtype=np.float32)
    
    if candidate_anchor_masks:
        active_seed_migs = set(seed_migs or [])
        
        intersection_count = np.zeros(len(verb_profiles), dtype=np.float32)
        union_count = np.zeros(len(verb_profiles), dtype=np.float32)
        
        for group, tags in MIGS.items():
            is_in_seed = group in active_seed_migs
            m = np.zeros(len(verb_profiles), dtype=bool)
            for t in tags:
                if t in candidate_anchor_masks:
                    m |= candidate_anchor_masks[t]
            
            intersection_count += (m & is_in_seed).astype(np.float32)
            union_count += (m | is_in_seed).astype(np.float32)

        # Add Artificial "Adult Only" Group (Steam Banner Flag)
        if mature_content_flags is not None:
            is_in_seed = bool(seed_mature_content)
            m = mature_content_flags.astype(bool)
            intersection_count += (m & is_in_seed).astype(np.float32)
            union_count += (m | is_in_seed).astype(np.float32)

        valid_union = union_count > 0
        skill_jaccard[valid_union] = intersection_count[valid_union] / union_count[valid_union]
        # Skill Jaccard is the differentiable core of mechanical similarity
        skill_multiplier = 0.2 + 0.8 * skill_jaccard

    # 2. Component Similarities
    # Mechanical Core (Verb Profiles * Skill Jaccard)
    ALPHA = 0.1
    intersection = np.sum(np.minimum(verb_profiles.astype(np.float32), seed_verb_profile.astype(np.float32)), axis=1)
    union = np.sum(np.maximum(verb_profiles.astype(np.float32), seed_verb_profile.astype(np.float32)), axis=1)
    denominator = intersection + ALPHA * (union - intersection)

    tag_sims = np.zeros(len(verb_profiles), dtype=np.float32)
    mask = denominator > 0
    tag_sims[mask] = intersection[mask] / denominator[mask]
    
    # Refine mechanical match with Skill Jaccard
    mechanical_match = tag_sims * skill_jaccard

    # Semantics (Double Normalized)
    sem_sims_raw = (np.dot(sem_vectors.astype(np.float32), seed_sem_vec.astype(np.float32)) /
                    (sem_norms + sem_lambda))
    sem_sims = sem_sims_raw / (seed_sem_norm + sem_lambda)

    # Topics (Standardized Cosine)
    fz_unit_seed = (seed_topic_dist.astype(np.float32) - topic_means) / (topic_stds + 1e-9)
    fz_unit_seed[fz_unit_seed < 0.0] = 0
    fn = np.linalg.norm(fz_unit_seed) + 1e-9
    fz_unit_seed = fz_unit_seed / fn

    topic_sims = np.zeros(len(verb_profiles), dtype=np.float32)
    batch_size = 100000
    for i in range(0, len(verb_profiles), batch_size):
        end = min(i + batch_size, len(verb_profiles))
        bz = (topic_distributions[i:end].astype(np.float32) - topic_means) / (topic_stds + 1e-9)
        bz[bz < 0.0] = 0
        bn = np.linalg.norm(bz, axis=1, keepdims=True) + 1e-9
        topic_sims[i:end] = np.dot(bz / bn, fz_unit_seed)

    # 3. Vibe Sim (Percentile Softmin)
    sem_cdf = pd.Series(sem_sims).rank(pct=True).values.astype(np.float32)
    topic_cdf = pd.Series(topic_sims).rank(pct=True).values.astype(np.float32)
    vibe_sim = softmin_blend([sem_cdf, topic_cdf], temperature=0.02)

    # 4. Total Similarity (Multi-Modal Blend)
    base_kernel = (0.10 * mechanical_match) + (0.90 * vibe_sim)
    
    # Apply Gaussian Spirit Filters (Tone and Difficulty)
    base_kernel *= tone_sim
    base_kernel *= diff_sim
    
    # 5. Apply Skill Barrier (Primary Gating)
    kernel = base_kernel * skill_multiplier

    # 6. Rescues / Penalties
    if precalculated_masks:
        if active_narrative_seed and len(active_narrative_seed) >= 2:
            match_counts = np.zeros(len(kernel), dtype=int)
            for t in active_narrative_seed:
                mask_key = t if t in precalculated_masks else f"tag_{t}"
                if mask_key in precalculated_masks:
                    match_counts += precalculated_masks[mask_key].astype(int)
            kernel += np.where(match_counts >= 3, 0.03, 0.0)
            vibe_sim = np.maximum(vibe_sim, np.where(match_counts >= 4, 0.01, 0.0))
            
        if is_cinematic_seed:
            # SOFT CINEMATIC VETO
            m_cin = candidate_anchor_masks.get("Cinematic", np.zeros(len(kernel), dtype=bool))
            m_sr = candidate_anchor_masks.get("Story Rich", np.zeros(len(kernel), dtype=bool))
            m_if = candidate_anchor_masks.get("Interactive Fiction", np.zeros(len(kernel), dtype=bool))
            
            # Narrative Strength (0.0 to 1.0 based on tag presence)
            narr_strength = (m_cin.astype(float) + m_sr.astype(float) + m_if.astype(float)) / 3.0
            # Smooth penalty that transitions based on narrative presence
            cin_penalty = 0.05 + 0.95 * soft_gate(narr_strength, threshold=0.1, temperature=0.05)
            kernel *= cin_penalty
            
            # Boost those that match the cinematic spirit perfectly
            cinematic_match = precalculated_masks.get("cinematic_resonance", np.zeros(len(kernel), dtype=bool))
            kernel += np.where(cinematic_match & (kernel > 0.02), 0.05, 0.0)
            
        if is_crpg_seed:
            i_mask = precalculated_masks.get("tag_Isometric", np.zeros(len(kernel), dtype=bool))
            c_mask = precalculated_masks.get("tag_CRPG", np.zeros(len(kernel), dtype=bool))
            kernel += np.where(i_mask & c_mask, 0.05, 0.0)

        # TITLE HIJACK PENALTY (Kept hard as it is a metadata correction)
        if "title_hijack" in precalculated_masks:
            hijack_mask = precalculated_masks["title_hijack"]
            kernel[hijack_mask] *= 0.1
            vibe_sim[hijack_mask] *= 0.1

    # 7. Soft Vetoes
    if candidate_anchor_masks:
        HORROR_MARKERS = {"Horror", "Survival Horror", "Psychological Horror", "Gore", "Violent"}
        seed_is_horror = any(t in HORROR_MARKERS for t in (seed_tags or [])) or any(t in HORROR_MARKERS for t in (active_narrative_seed or []))

        is_narrative_rescue = (active_narrative_seed is not None and len(active_narrative_seed) >= 2)
        # Rescue is now a probability weight
        rescue_weight = soft_gate(topic_sims, threshold=0.15, temperature=temperature)
        if is_narrative_rescue:
            rescue_weight = np.maximum(rescue_weight, 1.0)
        
        target_horror_count = np.zeros(len(kernel), dtype=int)
        for marker in HORROR_MARKERS:
            m_key = marker if marker in candidate_anchor_masks else f"tag_{marker}"
            if m_key in candidate_anchor_masks:
                target_horror_count += candidate_anchor_masks[m_key].astype(int)
        
        # Horror Clash is now a soft penalty
        horror_clash_prob = soft_gate(target_horror_count.astype(float), threshold=0.5, temperature=0.1) * (1.0 - float(seed_is_horror))
        rescue_weight *= (1.0 - horror_clash_prob)
        
        # Global Veto Mask (Identity Gating)
        if seed_migs:
            def calculate_soft_mig_clash(migs, masks):
                clash_count = np.zeros(len(kernel), dtype=float)
                for group_name in migs:
                    if group_name in MIGS:
                        m = np.zeros(len(kernel), dtype=bool)
                        for t in MIGS[group_name]:
                            m_key = t if t in masks else f"tag_{t}"
                            if m_key in masks: m |= masks[m_key].astype(bool)
                        clash_count += (~m).astype(float)
                
                # ADDITION: Conflicting MIG Veto (Identity Protection)
                # If seed lacks an 'Active' MIG that the candidate has, it's a clash
                ACTIVE_MIGS = {"SHOOTER", "FIGHTING", "MELEE_ACTION", "BULLET_HELL", "RACING", "SPORTS"}
                for group_name in ACTIVE_MIGS:
                    if group_name not in migs:
                        m = np.zeros(len(kernel), dtype=bool)
                        for t in MIGS[group_name]:
                            m_key = t if t in masks else f"tag_{t}"
                            if m_key in masks: m |= masks[m_key].astype(bool)
                        # Penalty for having a noisy MIG the seed lacks
                        # Increased to 1.0 (Full Clash) to enforce identity discipline
                        clash_count += (m.astype(float) * 1.0) 

                return clash_count / len(migs)

            mig_clash_score = calculate_soft_mig_clash(seed_migs, candidate_anchor_masks)
            
            # Hard Anchor Enforcement (Perspective, etc.)
            hard_clash_score = np.zeros(len(kernel), dtype=float)
            if seed_tags:
                hard_seed_tags = set(seed_tags) & HARD_ANCHORS
                if hard_seed_tags:
                    for t in hard_seed_tags:
                        m_key = t if t in candidate_anchor_masks else f"tag_{t}"
                        if m_key in candidate_anchor_masks:
                            hard_clash_score = np.maximum(hard_clash_score, (~candidate_anchor_masks[m_key].astype(bool)).astype(float))
            
            # Combine clashes: 
            # MIG clashes can be rescued by high thematic similarity
            # Hard clashes (Perspective, etc.) are IMMUNE to rescue
            veto_multiplier_mig = 0.001 + 0.999 * (1.0 - mig_clash_score * (1.0 - rescue_weight))
            veto_multiplier_hard = 0.001 + 0.999 * (1.0 - hard_clash_score)
            
            # Use the stricter of the two vetoes
            kernel *= np.minimum(veto_multiplier_mig, veto_multiplier_hard)

    # 8. Identity Intersection Rescues (Soul Matches)
    if candidate_anchor_masks and tone_z is not None and seed_tone_z is not None:
        seed_tags_all = set(seed_tags or []) | set(active_narrative_seed or [])
        SERIOUS = {"Education", "Math", "Science", "Typing", "Spelling", "Programming", "Logic"}
        seed_has_serious = any(t in SERIOUS for t in seed_tags_all)
        
        if seed_has_serious and seed_tone_z > 0.4:
            t_ser = np.zeros(len(kernel), dtype=int)
            for t in SERIOUS:
                m_key = t if t in candidate_anchor_masks else f"tag_{t}"
                if m_key in candidate_anchor_masks: t_ser += candidate_anchor_masks[m_key].astype(int)
            
            # Soul match if candidate has serious mechanics AND bizarre tone
            soul_match_prob = soft_gate(t_ser.astype(float), threshold=0.5) * soft_gate(tone_z, threshold=0.4)
            # Restore base score and apply boost smoothly
            kernel = kernel * (1.0 - soul_match_prob) + base_kernel * soul_match_prob
            kernel += soul_match_prob * 0.15

        # B. COGNITIVE DISSONANCE (Cute + Horror)
        CUTE = {"Cute", "Colorful", "Family Friendly", "Relaxing", "Anime"}
        HORROR = {"Horror", "Psychological Horror", "Survival Horror", "Gore", "Violent"}
        
        has_cut_seed = any(t in CUTE for t in seed_tags_all)
        has_hor_seed = any(t in HORROR for t in seed_tags_all)
        
        if has_cut_seed and has_hor_seed:
            t_cut = np.zeros(len(kernel), dtype=int)
            for t in CUTE:
                m_key = t if t in candidate_anchor_masks else f"tag_{t}"
                if m_key in candidate_anchor_masks: t_cut += candidate_anchor_masks[m_key].astype(int)
            
            t_hor = np.zeros(len(kernel), dtype=int)
            for t in HORROR:
                m_key = t if t in candidate_anchor_masks else f"tag_{t}"
                if m_key in candidate_anchor_masks: t_hor += candidate_anchor_masks[m_key].astype(int)
            
            soul_match_horror_prob = soft_gate(t_cut.astype(float), threshold=0.5) * soft_gate(t_hor.astype(float), threshold=0.5)
            kernel = kernel * (1.0 - soul_match_horror_prob) + base_kernel * soul_match_horror_prob
            kernel += soul_match_horror_prob * 0.15

    # 9. Soft Consensus Floor
    # Replaces hard kernel[vibe_sim < floor] = 0.0
    floor = 0.001 if (active_narrative_seed and len(active_narrative_seed) >= 2) else 0.005
    kernel *= soft_gate(vibe_sim, threshold=floor, temperature=0.001)
    
    final_kernel = np.maximum(kernel, 0.0)
    if return_components:
        return final_kernel, {'vibe': tag_sims, 'theme': sem_sims, 'cluster': topic_sims, 'combined': vibe_sim, 'difficulty': diff_sim, 'tone': tone_sim}
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
    sim = 1.0 - np.sqrt(np.maximum(js_div, 0))
    if mean is not None and std is not None:
        return (sim - mean) / (std if std > 1e-9 else 1.0)
    return sim
