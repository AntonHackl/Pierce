from benchmarks.common.pierce_timing import TIMING_POLICY, summarize_pierce_timing


def phase(ms):
    return {"duration_ms": ms}


def test_legacy_single_gpu_phases_are_summed():
    data = {
        "phases": {
            "Selectivity Estimation": phase(1.5),
            "Raytrace_Hash_Mesh1ToMesh2": phase(10.0),
            "Raytrace_Hash_Mesh2ToMesh1": phase(20.0),
            "compact_hash_table_pairs": phase(3.0),
            "Download Results": phase(4.0),
            "Plan Shared Slabs": phase(99.0),
        },
        "counters": {},
    }

    timing = summarize_pierce_timing(data, require_worker_counters=False)

    assert timing.query_time_ms == 38.5
    assert timing.breakdown["selectivity estimation"] == 1.5
    assert timing.breakdown["raytrace_hash_mesh1tomesh2"] == 10.0
    assert timing.overhead_breakdown["plan shared slabs"] == 99.0


def test_multi_gpu_overlap_uses_max_worker_not_sum():
    data = {
        "phases": {
            "Selectivity Estimation": phase(2.0),
            "Plan Shared Slabs": phase(100.0),
        },
        "counters": {
            "Profile_Active_GPU_Count": 2,
            "Profile_Slab_0_Worker_Measured_Hash_Query_Total_Us": 10_000,
            "Profile_Slab_0_Worker_Hash_Overlap_Mesh1_To_Mesh2_Us": 4_000,
            "Profile_Slab_0_Worker_Hash_Overlap_Mesh2_To_Mesh1_Us": 4_000,
            "Profile_Slab_0_Worker_Hash_Count_Table_Pairs_Us": 1_000,
            "Profile_Slab_0_Worker_Hash_Compact_Table_Us": 1_000,
            "Profile_Slab_1_Worker_Measured_Hash_Query_Total_Us": 20_000,
            "Profile_Slab_1_Worker_Hash_Overlap_Mesh1_To_Mesh2_Us": 8_000,
            "Profile_Slab_1_Worker_Hash_Overlap_Mesh2_To_Mesh1_Us": 8_000,
            "Profile_Slab_1_Worker_Hash_Count_Table_Pairs_Us": 2_000,
            "Profile_Slab_1_Worker_Hash_Compact_Table_Us": 2_000,
            "Profile_Global_Dedup_Total_Us": 3_000,
        },
    }

    timing = summarize_pierce_timing(data)

    assert timing.num_gpus_active == 2
    assert timing.query_time_ms == 25.0
    assert timing.breakdown["measured hash/raytrace query"] == 16.0
    assert timing.breakdown["hash compaction/result movement"] == 4.0
    assert timing.breakdown["global gpu deduplication"] == 3.0


def test_multi_gpu_intersection_includes_worker_downloads():
    data = {
        "phases": {
            "Selectivity Estimation": phase(2.0),
            "Initialize GPU Workers": phase(88.0),
        },
        "counters": {
            "Profile_Active_GPU_Count": 2,
            "Profile_Slab_0_Worker_Measured_Hash_Query_Total_Us": 20_000,
            "Profile_Slab_0_Worker_Download_Pairs_Us": 5_000,
            "Profile_Slab_1_Worker_Measured_Hash_Query_Total_Us": 15_000,
            "Profile_Slab_1_Worker_Download_Pairs_Us": 1_000,
            "Profile_Global_Dedup_Total_Us": 2_000,
        },
    }

    timing = summarize_pierce_timing(data)

    assert timing.query_time_ms == 29.0
    assert timing.breakdown["hash compaction/result movement"] == 5.0
    assert timing.overhead_breakdown["initialize gpu workers"] == 88.0


def test_missing_worker_counters_fail_for_new_steady_policy():
    data = {
        "phases": {
            "Selectivity Estimation": phase(2.5),
            "Execute Hash Query": phase(1000.0),
            "Global GPU Deduplication": phase(200.0),
            "Plan Shared Slabs": phase(3000.0),
        },
        "counters": {},
    }

    try:
        summarize_pierce_timing(data)
    except ValueError as exc:
        assert "requires Profile_Slab_*_Worker_Measured_Hash_Query_Total_Us counters" in str(exc)
    else:
        raise AssertionError("missing worker counters should fail for steady_query_v1")

    assert TIMING_POLICY == "steady_query_v1"
