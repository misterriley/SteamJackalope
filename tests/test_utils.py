import numpy as np
import pytest
from common.utils import to_z, calculate_hybrid_score
from common.constants import Z_SCORE_CLAMP_MIN, Z_SCORE_CLAMP_MAX

def test_calculate_hybrid_score_normalization():
    # Setup inputs
    z = np.array([1.0, 2.0])
    
    # Case 1: Weights sum to 1.0 (Identity)
    res = calculate_hybrid_score(
        z_semantic=z, w_semantic=1.0,
        z_tag=z, w_tag=0.0,
        z_spps=z, w_spps=0.0,
        z_date=z, w_date=0.0,
        z_pop=z, w_pop=0.0,
        z_length=z, w_length=0.0,
        z_difficulty=z, w_difficulty=0.0
    )
    assert np.allclose(res, z)

    # Case 2: Weights sum to 2.0. Result should be normalized.
    # Raw sum: z * 2.0. Total weight: 2.0. Result: z.
    res2 = calculate_hybrid_score(
        z_semantic=z, w_semantic=2.0,
        z_tag=z, w_tag=0.0,
        z_spps=z, w_spps=0.0,
        z_date=z, w_date=0.0,
        z_pop=z, w_pop=0.0,
        z_length=z, w_length=0.0,
        z_difficulty=z, w_difficulty=0.0
    )
    assert np.allclose(res2, z)

    # Case 3: Mixed weights
    # z=10. w_sem=1. z_tag=20. w_tag=1. Total weight=2. Raw sum=30. Result=15.
    res3 = calculate_hybrid_score(
        z_semantic=np.array([10.0]), w_semantic=1.0,
        z_tag=np.array([20.0]), w_tag=1.0,
        z_spps=np.array([0.0]), w_spps=0.0,
        z_date=np.array([0.0]), w_date=0.0,
        z_pop=np.array([0.0]), w_pop=0.0,
        z_length=np.array([0.0]), w_length=0.0,
        z_difficulty=np.array([0.0]), w_difficulty=0.0
    )
    assert res3[0] == 15.0
    
    # Case 4: Zero weights
    res4 = calculate_hybrid_score(
        z_semantic=z, w_semantic=0.0,
        z_tag=z, w_tag=0.0,
        z_spps=z, w_spps=0.0,
        z_date=z, w_date=0.0,
        z_pop=z, w_pop=0.0,
        z_length=z, w_length=0.0,
        z_difficulty=z, w_difficulty=0.0
    )
    assert np.all(res4 == 0.0)

def test_to_z_scaling():
    # Create data with high variance
    data = np.array([100.0, 0.0, 0.0, 0.0, 0.0])
    # mean = 20, std = 40
    # 100 -> (100-20)/40 = 2.0
    # 0 -> (0-20)/40 = -0.5
    z = to_z(data)
    
    assert z[0] == 2.0
    assert np.all(z[1:] == -0.5)
    
    # Check that an extreme outlier is NOT clamped (clamping moved to app)
    outlier_data = np.concatenate(([1000.0], np.zeros(100)))
    z_out = to_z(outlier_data)
    assert np.max(z_out) > Z_SCORE_CLAMP_MAX

def test_to_z_ignore_zeros():
    data = np.array([10.0, 10.0, 10.0, 0.0, 0.0, 0.0])
    # With ignore_zeros=True, subset is [10, 10, 10]. std is 0.
    # Logic: (x - mean) / 1.0 -> (10-10)/1 = 0.
    z = to_z(data, ignore_zeros=True)
    
    # Non-zeros should be 0 (mean)
    assert np.all(z[:3] == 0.0)
    
    # Zeros should remain -10.0
    # For the zero elements: (0 - 10) / 1 = -10.
    assert np.all(z[3:] == -10.0)

    # Test distinct values
    data2 = np.array([10.0, 20.0, 0.0, 0.0])
    # subset [10, 20]. mean 15. std 5.
    # 10 -> (10-15)/5 = -1
    # 20 -> (20-15)/5 = 1
    # 0 -> (0-15)/5 = -3
    z2 = to_z(data2, ignore_zeros=True)
    assert z2[0] == -1.0
    assert z2[1] == 1.0
    assert z2[2] == -3.0

def test_to_z_all_zeros():
    data = np.zeros(10)
    z = to_z(data, ignore_zeros=True)
    assert np.all(z == 0)
