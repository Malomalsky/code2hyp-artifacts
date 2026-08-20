from scripts.analyze_codenet_java_stage_b_power import simulate_location_shift_power


def test_power_is_deterministic_and_monotone_in_effect_and_variance() -> None:
    kwargs = {
        "cluster_count": 40,
        "simulations": 10_000,
        "rng_seed": 7,
        "effects": (0.1, 0.2),
        "variance_scales": (1.0, 2.0),
    }
    first = simulate_location_shift_power((-1.0, -0.5, 0.5, 1.0), **kwargs)
    second = simulate_location_shift_power((-1.0, -0.5, 0.5, 1.0), **kwargs)
    assert first == second
    by_key = {(row["variance_scale"], row["true_location_shift"]): row for row in first}
    assert by_key[(1.0, 0.2)]["marginal_power"] > by_key[(1.0, 0.1)]["marginal_power"]
    assert by_key[(1.0, 0.1)]["marginal_power"] > by_key[(2.0, 0.1)]["marginal_power"]
