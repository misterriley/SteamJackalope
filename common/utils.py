import numpy as np
import re
from common.constants import EPSILON, Z_SCORE_CLAMP_MIN, Z_SCORE_CLAMP_MAX, SOFTMIN_TEMPERATURE

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
    This is the single source of truth for both the Solver preview and Backend recommender.
    """
    # 1. Apply Clamping to Metadata
    q = np.clip(z_quality, z_clamp_min, z_clamp_max)
    d = np.clip(z_date, z_clamp_min, z_clamp_max)
    p = np.clip(z_pop, z_clamp_min, z_clamp_max)
    l = np.clip(z_playtime, z_clamp_min, z_clamp_max)
    diff = np.clip(z_difficulty, z_clamp_min, z_clamp_max)
    pr = np.clip(z_price, z_clamp_min, z_clamp_max)
    
    # 2. DNA Components (Linear Aggregate)
    # The Taste DNA profile is a linear sum of your aggregate preferences.
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
    # These use 3-mode consensus (Hard Softmin) to prevent keyword hijacking.
    # We use T=0.05 to enforce an "AND" relationship.
    # We only apply consensus if the modalities are active (> 0).
    
    seed_contrib = 0.0
    if seed_tag_sim is not None:
        # Build consensus signals
        signals = []
        # If weight is 0, the mode is effectively disabled (it can't veto)
        if weights.get('tag_match', 0.0) > 1e-9: signals.append(seed_tag_sim)
        if w_semantic > 1e-9 and seed_sem_sim is not None: signals.append(seed_sem_sim)
        if w_topic > 1e-9 and seed_topic_sim is not None: signals.append(seed_topic_sim)
        
        if signals:
            consensus_sim = softmin_blend(signals, temperature=0.05)
            # Apply the average of the active thematic weights as a master multiplier
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
    from common.constants import CHI_FIT_NORM_THRESHOLD, CHI_FIT_PERCENTILE
    from scipy.stats import chi
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
    Signals are expected to be NumPy arrays of the same shape.
    This creates 'consensus' logic: the result is heavily weighted towards the lowest score.
    """
    if not signals:
        return 0.0
    if len(signals) == 1:
        return signals[0]
        
    stack = np.stack(signals, axis=0) # (num_signals, num_games)
    
    # Calculate weights: w_i = exp(-s_i / T) / sum(exp(-s_j / T))
    # Using log-sum-exp trick for stability
    scaled = -stack / temperature
    max_val = np.max(scaled, axis=0)
    exp_vals = np.exp(scaled - max_val)
    weights = exp_vals / np.sum(exp_vals, axis=0)
    
    # Blended similarity: sum(s_i * w_i)
    return np.sum(stack * weights, axis=0)

def calculate_jackalope_kernel(
    tag_vectors, tag_norms, seed_tag_vec, seed_tag_norm,
    sem_vectors, sem_norms, seed_sem_vec, seed_sem_norm,
    topic_distributions, seed_topic_dist,
    topic_means, topic_stds,
    tag_scaling_factor, dot_product_lambda,
    sem_scaling_factor, sem_lambda,
    topic_scaling_factor=0.1,
    # Veto/Rescue Metadata
    seed_anchors=None,
    candidate_anchor_masks=None, # dict: anchor_name -> boolean mask
    active_narrative_seed=None,
    is_cinematic_seed=False,
    seed_has_heavy_action=False,
    action_tags=None,
    mystery_tags=None,
    is_mystery_seed=False,
    rpg_action_tags=None,
    rpg_crpg_tags=None,
    is_action_rpg_seed=False,
    is_crpg_seed=False,
    loop_tags=None,
    is_loop_seed=False,
    is_horror_seed=False,
    full_tags_series=None, # pd.Series of stringified tags for regex rescues
    # PRE-CALCULATED MASKS (Optimization)
    precalculated_masks=None, # dict: key -> boolean mask
    return_components=False
):
    """
    The Jackalope Kernel: A multi-modal 'Mechanical Identity' measure.
    Combines Tags, Semantics, and Topics with structural vetoes.
    """
    # 1. Component Similarities
    # Tags
    tag_sims = (np.dot(tag_vectors.astype(np.float32), seed_tag_vec.astype(np.float32)) / 
                ((tag_norms + dot_product_lambda) * (seed_tag_norm + dot_product_lambda))) * tag_scaling_factor
    
    # Semantics (Double Normalized)
    sem_sims_raw = (np.dot(sem_vectors.astype(np.float32), seed_sem_vec.astype(np.float32)) / 
                    (sem_norms + sem_lambda)) * sem_scaling_factor
    sem_sims = sem_sims_raw / (seed_sem_norm + sem_lambda)
    
    # Topics (ReLU-Standardized Cosine)
    fz = (seed_topic_dist.astype(np.float32) - topic_means) / (topic_stds + 1e-9)
    fz[fz < 2.5] = 0
    fn = np.linalg.norm(fz) + 1e-9
    fz_unit = fz / fn
    
    # Batch process topics if large
    topic_sims = np.zeros(len(tag_norms), dtype=np.float32)
    batch_size = 100000 
    for i in range(0, len(tag_norms), batch_size):
        end = min(i + batch_size, len(tag_norms))
        bz = (topic_distributions[i:end].astype(np.float32) - topic_means) / (topic_stds + 1e-9)
        bz[bz < 2.5] = 0
        bn = np.linalg.norm(bz, axis=1, keepdims=True) + 1e-9
        topic_sims[i:end] = np.dot(bz / bn, fz_unit)
        
    # Rescue peaks (relaxed)
    rescue_mask = (np.maximum(tag_sims, 0) + np.maximum(sem_sims, 0)) > 0.05
    if np.any(rescue_mask):
        fz_low = (seed_topic_dist.astype(np.float32) - topic_means) / (topic_stds + 1e-9)
        fz_low[fz_low < 1.5] = 0
        fn_low = np.linalg.norm(fz_low) + 1e-9
        fz_low_unit = fz_low / fn_low
        
        rescue_indices = np.where(rescue_mask)[0]
        for i in range(0, len(rescue_indices), batch_size):
            batch_indices = rescue_indices[i:i+batch_size]
            bz = (topic_distributions[batch_indices].astype(np.float32) - topic_means) / (topic_stds + 1e-9)
            bz[bz < 1.5] = 0
            bn = np.linalg.norm(bz, axis=1, keepdims=True) + 1e-9
            topic_sims[batch_indices] = np.maximum(topic_sims[batch_indices], np.dot(bz / bn, fz_low_unit))

    # 2. Hard Softmin Consensus
    consensus_sim = softmin_blend([tag_sims, sem_sims, topic_sims * topic_scaling_factor], temperature=0.01)
    
    # 3. Hierarchical Blend
    kernel = (tag_sims * 0.25 + sem_sims * 0.25 + consensus_sim * 0.5)
    
    # 4. Rescues
    if precalculated_masks:
        if active_narrative_seed and len(active_narrative_seed) >= 2:
            match_counts = np.zeros(len(kernel), dtype=int)
            for t in active_narrative_seed:
                mask_key = f"tag_{t}"
                if mask_key in precalculated_masks:
                    match_counts += precalculated_masks[mask_key].astype(int)
            kernel += np.where(match_counts >= 3, 0.03, 0.0)
            consensus_sim = np.maximum(consensus_sim, np.where(match_counts >= 4, 0.01, 0.0))
            
        if is_cinematic_seed:
            cinematic_match = precalculated_masks.get("cinematic_resonance", np.zeros(len(kernel), dtype=bool))
            kernel += np.where(cinematic_match & (kernel > 0.02), 0.05, 0.0)
            
        if is_crpg_seed:
            i_mask = precalculated_masks.get("tag_Isometric", np.zeros(len(kernel), dtype=bool))
            c_mask = precalculated_masks.get("tag_CRPG", np.zeros(len(kernel), dtype=bool))
            kernel += np.where(i_mask & c_mask, 0.05, 0.0)

    # 5. Vetoes
    if candidate_anchor_masks:
        # AUTOMATIC HORROR DETECTION (Strict Tag-based)
        HORROR_MARKERS = {"Horror", "Survival Horror", "Psychological Horror", "Gore", "Violent"}
        seed_is_horror = False
        if seed_anchors and any(t in HORROR_MARKERS for t in seed_anchors):
            seed_is_horror = True
        elif active_narrative_seed and any(t in HORROR_MARKERS for t in active_narrative_seed):
            seed_is_horror = True

        # STYLISTIC RESCUE CALCULATION (Master Bypass)
        # Rescue if Topic similarity is high (> 0.15)
        # or if both games are Narrative-heavy (at least 2 markers).
        is_narrative_rescue = (active_narrative_seed is not None and len(active_narrative_seed) >= 2)
        rescue_mask = (topic_sims > 0.15) | is_narrative_rescue
        
        # Anti-Horror Clash Protection (Strict)
        # If target has ANY horror markers and seed has NONE, block the rescue.
        target_horror_count = np.zeros(len(kernel), dtype=int)
        for marker in HORROR_MARKERS:
            if marker in candidate_anchor_masks:
                target_horror_count += candidate_anchor_masks[marker].astype(int)
        
        is_horror_clash = (target_horror_count >= 1) & ~seed_is_horror
        rescue_mask = rescue_mask & ~is_horror_clash # Absolute block for clashes
        
        # NEW: Mechanical Conflict Protection
        # If seed is RPG/Roguelike but target is Platformer (or vice versa), block the rescue.
        PLATFORMER_VARIANTS = {"Platformer", "2D Platformer", "3D Platformer", "Precision Platformer", "Puzzle Platformer"}
        
        is_platformer_target = np.zeros(len(kernel), dtype=bool)
        for pv in PLATFORMER_VARIANTS:
            if pv in candidate_anchor_masks:
                is_platformer_target |= candidate_anchor_masks[pv]
        
        is_rpg_seed = "RPG" in seed_anchors
        is_platformer_seed = any(pv in seed_anchors for pv in PLATFORMER_VARIANTS)
        
        # Broad Roguelike Detection
        ROGUE_MARKERS = {"Roguelike", "Roguelite", "Action Roguelike", "Bullet Hell"}
        is_rogue_seed = any(t in ROGUE_MARKERS for t in seed_anchors) if seed_anchors else False
        
        # Conflict check: block stylistic rescue for mechanical mismatches
        is_rpg_target = (("RPG" in candidate_anchor_masks and candidate_anchor_masks["RPG"]) | \
                         ("Roguelike" in candidate_anchor_masks and candidate_anchor_masks["Roguelike"]) | \
                         ("Roguelite" in candidate_anchor_masks and candidate_anchor_masks["Roguelite"]) | \
                         ("Action Roguelike" in candidate_anchor_masks and candidate_anchor_masks["Action Roguelike"]))
        
        # Exception: Allow RPG seeds to match Platformer targets if the target IS also an RPG (Action RPG Platformers)
        is_invalid_platformer_overlap = is_platformer_target & ~is_rpg_target
        
        is_mechanical_clash = (is_rpg_seed & is_invalid_platformer_overlap) | \
                              (is_rogue_seed & is_platformer_target) | \
                              (is_platformer_seed & is_rpg_target)
        
        # Strategy vs. Non-Strategy Action/Survival conflict
        is_strategy_target = candidate_anchor_masks.get("Strategy", np.zeros(len(kernel), dtype=bool))
        is_strategy_seed = "Strategy" in seed_anchors if seed_anchors else False
        is_survival_seed = "Survival" in seed_anchors if seed_anchors else False
        
        if not is_strategy_seed and (is_rpg_seed or is_rogue_seed or is_platformer_seed or is_survival_seed):
            # Exception: RPG/JRPG seeds matching Strategy targets that are ALSO RPGs (Tactical RPGs)
            is_rpg_target = candidate_anchor_masks.get("RPG", np.zeros(len(kernel), dtype=bool)) | \
                            candidate_anchor_masks.get("JRPG", np.zeros(len(kernel), dtype=bool))
            is_valid_rpg_overlap = (is_rpg_seed or "JRPG" in seed_anchors) & is_rpg_target
            
            is_mechanical_clash |= (is_strategy_target & ~is_valid_rpg_overlap)
        
        # MANDATORY BLOCK: Precision Platformers are too mechanically unique to transcend.
        is_precision_target = candidate_anchor_masks.get("Precision Platformer", np.zeros(len(kernel), dtype=bool))
        if not is_platformer_seed:
            is_mechanical_clash |= is_precision_target
            
        rescue_mask = rescue_mask & ~is_mechanical_clash # Absolute block for clashes

        # Build Veto Mask
        global_veto_mask = np.zeros(len(kernel), dtype=bool)
        HARD_ANCHORS = {
            "Platformer", "Puzzle", "Strategy", "RPG", "Roguelike", "Souls-like", "Metroidvania", 
            "JRPG", "Survival", "Visual Novel", "FPS", "Third Person", "Shooter",
            "Turn-Based Combat", "Turn-Based Strategy", "Turn-Based Tactics",
            "Hack and Slash", "Spectacle fighter"
        }
        
        if seed_anchors:
            # PERSPECTIVE RULE: FPS/Third Person are only HARD if the seed is a Shooter.
            is_shooter_seed = "Shooter" in seed_anchors or "Looter Shooter" in seed_anchors
            PERSPECTIVE_GENRES = {"FPS", "Third Person"}
            
            # MECHANICAL GROUPINGS
            TURN_BASED_GROUP = {"Turn-Based Combat", "Turn-Based Strategy", "Turn-Based Tactics"}
            ACTION_COMBAT_GROUP = {"Hack and Slash", "Spectacle fighter"}
            
            # ACTION vs TURN-BASED detection for clashes
            is_turn_based_seed = any(a in TURN_BASED_GROUP for a in seed_anchors)
            is_action_realtime_seed = any(a in (ACTION_COMBAT_GROUP | {"FPS", "Shooter"}) for a in seed_anchors)
            
            # Perspective/Shooter conflict detection
            is_fps_seed = "FPS" in seed_anchors or "First-Person" in seed_anchors
            is_tps_seed = "Third Person" in seed_anchors or "TPS" in seed_anchors
            
            is_fps_target = candidate_anchor_masks.get("FPS", np.zeros(len(kernel), dtype=bool)) | \
                            candidate_anchor_masks.get("First-Person", np.zeros(len(kernel), dtype=bool))
            is_tps_target = candidate_anchor_masks.get("Third Person", np.zeros(len(kernel), dtype=bool)) | \
                            candidate_anchor_masks.get("TPS", np.zeros(len(kernel), dtype=bool))
            is_shooter_target = candidate_anchor_masks.get("Shooter", np.zeros(len(kernel), dtype=bool)) | \
                                candidate_anchor_masks.get("Looter Shooter", np.zeros(len(kernel), dtype=bool))

            perspective_clash = np.zeros(len(kernel), dtype=bool)
            if is_shooter_seed:
                perspective_clash = (is_fps_seed & is_tps_target) | (is_tps_seed & is_fps_target)
            
            shooter_clash = (is_shooter_seed & ~is_shooter_target)
            
            is_turn_based_target = np.zeros(len(kernel), dtype=bool)
            for a in TURN_BASED_GROUP:
                if a in candidate_anchor_masks:
                    is_turn_based_target |= candidate_anchor_masks[a]
            
            action_clash = (is_action_realtime_seed & is_turn_based_target) | (is_turn_based_seed & ~is_turn_based_target & is_action_realtime_seed)
            is_mechanical_clash |= perspective_clash | shooter_clash | action_clash

            # Anime stylistic lock
            is_anime_seed = "Anime" in seed_anchors
            is_anime_target = candidate_anchor_masks.get("Anime", np.zeros(len(kernel), dtype=bool))
            if is_anime_seed:
                rescue_mask = rescue_mask & is_anime_target.astype(bool)

            def calculate_anchor_veto(anchors, masks):
                veto = np.zeros(len(kernel), dtype=bool)
                for a in anchors:
                    if a in masks:
                        if a in PERSPECTIVE_GENRES and not is_shooter_seed:
                            continue
                        if a == "JRPG":
                            has_rpg_target = masks.get("RPG", np.zeros(len(kernel), dtype=bool))
                            has_anime_target = masks.get("Anime", np.zeros(len(kernel), dtype=bool))
                            rescue_jrpg = (has_rpg_target.astype(bool) & has_anime_target.astype(bool))
                            veto |= (~masks[a].astype(bool) & ~rescue_jrpg)
                        elif a in TURN_BASED_GROUP:
                            target_has_any_tb = np.zeros(len(kernel), dtype=bool)
                            for tb in TURN_BASED_GROUP:
                                if tb in masks:
                                    target_has_any_tb |= masks[tb]
                            veto |= ~target_has_any_tb
                        elif a in ACTION_COMBAT_GROUP:
                            target_has_any_ac = np.zeros(len(kernel), dtype=bool)
                            for ac in ACTION_COMBAT_GROUP:
                                if ac in masks:
                                    target_has_any_ac |= masks[ac]
                            veto |= ~target_has_any_ac
                        else:
                            veto |= ~masks[a].astype(bool)
                return veto

            # Layer 1: Rescuable Vetoes
            global_veto_mask = calculate_anchor_veto(seed_anchors, candidate_anchor_masks)
            global_veto_mask &= ~rescue_mask.astype(bool)
            
            # Layer 2: Mandatory Vetoes (Clashes and Hard Genres)
            global_veto_mask |= is_mechanical_clash.astype(bool)
            seed_hard_anchors = [a for a in seed_anchors if a in HARD_ANCHORS]
            global_veto_mask |= calculate_anchor_veto(seed_hard_anchors, candidate_anchor_masks)
        
        # NEW: Low Consensus Floor
        # If Topic similarity is weak and they aren't rescued, they shouldn't be matched.
        is_low_consensus = ((topic_sims < 0.15) & ~rescue_mask.astype(bool)).astype(bool)
        global_veto_mask |= is_low_consensus
        
        # Apply the multiplier once
        kernel[global_veto_mask] *= 0.001
                    
        if precalculated_masks:
            if not seed_has_heavy_action:
                early_action_mask = precalculated_masks.get("early_action", np.zeros(len(kernel), dtype=bool))
                kernel[early_action_mask] *= 0.01
                
            if is_mystery_seed:
                mystery_mask = precalculated_masks.get("mystery_any", np.zeros(len(kernel), dtype=bool))
                soul_rescue = precalculated_masks.get("soul_rescue", np.zeros(len(kernel), dtype=bool))
                kernel[~mystery_mask & ~soul_rescue] *= 0.01
                
            if is_action_rpg_seed:
                crpg_m = candidate_anchor_masks.get("CRPG", precalculated_masks.get("tag_CRPG", np.zeros(len(kernel), dtype=bool)))
                act_m = candidate_anchor_masks.get("Action RPG", precalculated_masks.get("tag_Action RPG", np.zeros(len(kernel), dtype=bool)))
                kernel[crpg_m & ~act_m] *= 0.05
            if is_crpg_seed:
                act_m = candidate_anchor_masks.get("Action RPG", precalculated_masks.get("tag_Action RPG", np.zeros(len(kernel), dtype=bool)))
                crpg_m = candidate_anchor_masks.get("CRPG", precalculated_masks.get("tag_CRPG", np.zeros(len(kernel), dtype=bool)))
                kernel[act_m & ~crpg_m] *= 0.05

            if not is_loop_seed:
                loop_mask = precalculated_masks.get("loop_any", np.zeros(len(kernel), dtype=bool))
                kernel[loop_mask] *= 0.01
                
            if not is_horror_seed:
                horror_mask = precalculated_masks.get("horror_any", np.zeros(len(kernel), dtype=bool))
                nar_mask = precalculated_masks.get("story_rich_or_choices", np.zeros(len(kernel), dtype=bool))
                kernel[horror_mask & ~nar_mask] *= 0.1

    # 6. Consensus Floor Veto
    if active_narrative_seed and len(active_narrative_seed) >= 2:
        kernel[consensus_sim < 0.001] = 0.0
    else:
        kernel[consensus_sim < 0.005] = 0.0
    
    # NEW: Hard Veto for extreme component mismatch (The 'Ghost Match' Protection)
    # If one modality is massive but the other is weak, it's keyword/topic noise.
    # 1. Absolute Floor: Semantic must be at least 0.05 if Tag is high.
    kernel[(tag_sims > 0.5) & (sem_sims < 0.05)] = 0.0
    # 2. Relative Veto: If Tag sim is > 10x Semantic sim, it's a 'Ghost Match'
    kernel[(tag_sims > 0.2) & (tag_sims > (sem_sims * 10.0))] = 0.0
    # 3. Inverse Check
    kernel[(sem_sims > 0.5) & (tag_sims < 0.05)] = 0.0
        
    final_kernel = np.maximum(kernel, 0.0)
    
    if return_components:
        return final_kernel, {
            'vibe': tag_sims,
            'theme': sem_sims,
            'cluster': topic_sims,
            'combined': consensus_sim
        }
        
    return final_kernel

def fast_jsd_similarity(p, Q_matrix, mean=None, std=None):
    """
    Calculates Jensen-Shannon Similarity (1 - JSD) between a probability vector p 
    and a matrix of distributions Q. Optionally Z-scores the result using provided 
    population mean and std.
    """
    eps = 1e-12
    # Ensure p and Q are normalized and strictly positive for log
    p = np.clip(p, eps, 1.0)
    Q = np.clip(Q_matrix, eps, 1.0)
    # Re-normalize to 1.0
    p = p / np.sum(p)
    Q = Q / np.sum(Q, axis=-1, keepdims=True)
    
    m = 0.5 * (p + Q)
    
    # JS Div = 0.5 * KLD(P||M) + 0.5 * KLD(Q||M)
    # Using np.where to avoid log(0) and maintaining precision
    term1 = p * np.log(p / m)
    term2 = Q * np.log(Q / m)
    js_div = 0.5 * (np.sum(term1, axis=-1) + np.sum(term2, axis=-1))
    
    # Similarity = 1 - sqrt(JSD)
    # JSD is strictly [0, ln(2)] for base e or [0, 1] for base 2. 
    # Our formula uses natural log, so we clamp to ln(2)
    sim = 1.0 - np.sqrt(np.maximum(js_div, 0))
    
    if mean is not None and std is not None:
        # Z-score the similarity based on population stats
        return (sim - mean) / (std if std > 1e-9 else 1.0)
    
    return sim
