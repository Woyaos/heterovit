"""Focused correctness checks for dependency-certified FFN partitioning."""

import copy
import importlib.util
import itertools
import json
import math
import unittest
from pathlib import Path


HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("auto_block_partition", HERE / "partition.py")
partition = importlib.util.module_from_spec(spec)
spec.loader.exec_module(partition)


def exhaustive_cost(region, tasks_by_id, model, capability, runtime):
    """Enumerate raw placements plus the declared whole-FFN implementation."""
    task_ids = region["task_ids"]
    tasks = [tasks_by_id[task_id] for task_id in task_ids]
    edges = [region["input_edge"], *region["internal_edges"], region["output_edge"]]
    costs = []
    for devices in itertools.product(("GPU", "FPGA"), repeat=len(tasks)):
        candidate = [partition.candidate_cost([task], device, capability, model, runtime)[0]
                     for task, device in zip(tasks, devices)]
        if any(value is None for value in candidate):
            continue
        chain = ("GPU", *devices, "GPU")
        costs.append(sum(candidate) + sum(
            partition.transfer_us(edge, chain[i], chain[i + 1], model)
            for i, edge in enumerate(edges)
        ))
    full, _ = partition.candidate_cost(tasks, "FPGA", capability, model, runtime)
    if full is not None:
        costs.append(full + partition.transfer_us(edges[0], "GPU", "FPGA", model)
                     + partition.transfer_us(edges[-1], "FPGA", "GPU", model))
    return min(costs)


class PartitionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.runtime = partition.load_runtime()
        cls.dag = json.loads(partition.DAG_PATH.read_text(encoding="utf-8"))
        cls.profile = json.loads(partition.PROFILE_PATH.read_text(encoding="utf-8"))
        cls.capability = json.loads(partition.CAPABILITY_PATH.read_text(encoding="utf-8"))
        cls.tasks = {task["id"]: task for task in cls.dag["tasks"]}

    def test_recognizes_twelve_branch_free_ffns(self):
        regions, rejected = partition.ffns_from_dependencies(self.dag)
        self.assertEqual(12, len(regions))
        self.assertEqual([], rejected)
        self.assertEqual(36, len({task_id for r in regions for task_id in r["task_ids"]}))

    def test_dp_matches_exhaustive_under_multiple_profiles(self):
        regions, _ = partition.ffns_from_dependencies(self.dag)
        for speedup, fixed_us, fused_supported in itertools.product(
                (1.0, 4.0, 12.0), (25.0, 250.0), (True, False)):
            profile = copy.deepcopy(self.profile)
            profile["fpga"]["linear_speedup_over_gpu"] = speedup
            profile["communication_path"]["host_fpga_pcie"]["fixed_latency_us"] = fixed_us
            capability = copy.deepcopy(self.capability)
            if not fused_supported:
                capability["supported_fpga_blocks"] = []
            model = self.runtime.SensitivityCostModel(profile)
            for region in regions:
                solved = partition.solve_region(region, self.tasks, model, capability, self.runtime)
                brute = exhaustive_cost(region, self.tasks, model, capability, self.runtime)
                self.assertTrue(math.isclose(brute, solved["interval_cost_us"], abs_tol=1e-6),
                                (speedup, fixed_us, fused_supported, region["id"]))

    def test_workspace_limit_disables_fpga_without_disabling_gpu(self):
        capability = copy.deepcopy(self.capability)
        capability["max_workspace_bytes"] = 1
        regions, _ = partition.ffns_from_dependencies(self.dag)
        model = self.runtime.SensitivityCostModel(self.profile)
        for region in regions:
            solution = partition.solve_region(region, self.tasks, model, capability, self.runtime)
            self.assertEqual({"GPU"}, {block["device"] for block in solution["blocks"]})

    def test_nonuniform_costs_select_only_eleven_full_ffns(self):
        regions, _ = partition.ffns_from_dependencies(self.dag)
        profile = copy.deepcopy(self.profile)
        profile["task_costs_us"] = {
            task_id: {"GPU": 100.0} for task_id in regions[0]["task_ids"]
        }
        result = partition.run(self.dag, profile, self.capability, self.runtime)
        full_fpga = [solution for solution in result["regions"]
                     if len(solution["blocks"]) == 1
                     and solution["blocks"][0]["device"] == "FPGA"]
        self.assertEqual(11, len(full_fpga))
        self.assertLess(result["comparison"]["automatic_region_dp"]["makespan_us"],
                        result["comparison"]["fixed_ffn"]["makespan_us"])

    def test_branching_ffn_is_not_fused(self):
        dag = copy.deepcopy(self.dag)
        regions, _ = partition.ffns_from_dependencies(dag)
        fc1 = regions[0]["task_ids"][0]
        extra = copy.deepcopy(regions[0]["internal_edges"][0])
        extra["target"] = regions[0]["output_edge"]["target"]
        dag["edges"].append(extra)
        regions_after, rejected = partition.ffns_from_dependencies(dag)
        self.assertEqual(11, len(regions_after))
        self.assertTrue(any(item["root"] == fc1 for item in rejected))

    def test_quotient_keeps_external_edges_and_once_only_compute(self):
        regions, _ = partition.ffns_from_dependencies(self.dag)
        model = self.runtime.SensitivityCostModel(self.profile)
        solutions = [partition.solve_region(r, self.tasks, model, self.capability, self.runtime)
                     for r in regions]
        fused, replacements = partition.quotient_dag(self.dag, solutions, self.runtime)
        self.assertEqual(125 - 2 * len(replacements) // 3, len(fused["tasks"]))
        for edge in self.dag["edges"]:
            source = replacements.get(edge["source"], edge["source"])
            target = replacements.get(edge["target"], edge["target"])
            if source != target:
                self.assertTrue(any(e["source"] == source and e["target"] == target
                                    for e in fused["edges"]))
        placements = partition.fixed_placements(fused, solutions, replacements)
        replay = partition.evaluate_fixed(fused, placements, model, self.runtime)
        self.assertEqual(len(fused["tasks"]), len(replay["task_timeline"]))
        self.assertEqual(len(replay["task_timeline"]), len(set(
            item["task_id"] for item in replay["task_timeline"])))

    def test_synthetic_complete_target_profile_uses_only_measured_feasible_costs(self):
        regions, _ = partition.ffns_from_dependencies(self.dag)
        source = self.runtime.SensitivityCostModel(self.profile)
        costs = {task["id"]: {"GPU": source.latency_us(task, "GPU")}
                 for task in self.dag["tasks"]}
        for region in regions:
            components = [self.tasks[task_id] for task_id in region["task_ids"]]
            for task in (components[0], components[2]):
                costs[task["id"]]["FPGA"] = source.latency_us(task, "FPGA")
            island = partition.block_task(components, self.runtime)
            costs[island["id"]] = {"FPGA": source.latency_us(island, "FPGA")}
        target = copy.deepcopy(self.profile)
        target["profile_kind"] = "target_measurement"
        target["status"] = "complete"
        target["valid_for_target_prediction"] = True
        target["profile_name"] = "synthetic_test_fixture_not_hardware_data"
        target["task_costs_us"] = costs
        target.pop("gpu")
        target.pop("fpga")
        verified_fixture_capability = copy.deepcopy(self.capability)
        verified_fixture_capability["status"] = "verified_target_capability"
        with self.assertRaisesRegex(RuntimeError, "verified FPGA capability"):
            partition.run(self.dag, target, self.capability, self.runtime)
        result = partition.run(self.dag, target, verified_fixture_capability, self.runtime)
        self.assertIn("static_declared_singletons_fpga", result["comparison"])
        self.assertNotIn("static_all_linear_fpga", result["comparison"])
        self.assertEqual(48, result["comparison"]["static_declared_singletons_fpga"]
                         ["cross_device_edges"])
        incomplete = copy.deepcopy(target)
        del incomplete["task_costs_us"]["block_00_ffn_island"]["FPGA"]
        with self.assertRaisesRegex(RuntimeError, "Missing measured FPGA cost"):
            partition.run(self.dag, incomplete, verified_fixture_capability, self.runtime)


if __name__ == "__main__":
    unittest.main()
