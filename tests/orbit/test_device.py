"""
Device facts in the optimizer's context (plan §9.5, §11.7).

Written after an agent proposed BLOCK_SIZE=8192 and num_warps=16 for a device with a
1024 work-group limit and 16 EUs. Both proposals were well argued and both measured
roughly 2x slower, because nothing had told the agent what machine it was optimizing.
"""

from __future__ import annotations

import pytest

from xe_forge.orbit.device import (
    SMALL_DEVICE_EU_THRESHOLD,
    DeviceFacts,
    launch_constraints,
    probe_device,
)


def _wildcat() -> DeviceFacts:
    """The machine this was found on, as measured."""
    return DeviceFacts(
        name="Intel(R) Graphics",
        platform="Level-Zero V2",
        driver_version="1.17.39395",
        integrated=True,
        eu_count=16,
        compute_units=16,
        subslices=2,
        max_work_group_size=1024,
        sub_group_sizes=[16, 32],
        local_mem_bytes=131072,
        last_level_cache_bytes=2097152,
        total_memory_bytes=14862204928,
        memory_bus_width=64,
        has_fp16=True,
        has_bf16=True,
        has_fp64=True,
        has_matrix_engine=True,
        available=True,
    )


class TestUnknownIsNotDefaulted:
    """A plausible default for the popular device is what produced the wrong advice."""

    def test_an_unavailable_device_says_so_rather_than_inventing_one(self):
        text = DeviceFacts().describe()
        assert "unknown" in text
        assert "untethered from the hardware" in text

    def test_an_unavailable_device_yields_no_constraints(self):
        assert launch_constraints(DeviceFacts()) == []

    def test_probing_without_torch_or_a_device_is_not_an_error(self):
        facts = probe_device()
        assert isinstance(facts, DeviceFacts)


class TestHardLimitsLeadTheDescription:
    def test_the_work_group_ceiling_is_stated_as_a_hard_limit(self):
        """A block larger than this cannot launch — that invalidates, not just slows."""
        text = _wildcat().describe()
        assert "HARD LIMIT" in text
        assert "1024" in text

    def test_sub_group_sizes_are_plural_not_a_single_assumed_width(self):
        """The agent asserted 'Intel XPU uses a 16-wide sub-group'; this one supports both."""
        text = _wildcat().describe()
        assert "16, 32" in text
        assert "not a single fixed width" in text

    def test_integration_and_eu_count_are_reported(self):
        text = _wildcat().describe()
        assert "integrated" in text
        assert "16 EUs" in text

    def test_shared_memory_is_flagged_on_an_integrated_device(self):
        assert "shared with the host" in _wildcat().describe()


class TestSmallDeviceWarning:
    def test_a_small_device_warns_that_discrete_advice_does_not_transfer(self):
        text = _wildcat().describe()
        assert "does not transfer" in text
        assert "oversubscribe" in text

    def test_a_large_device_gets_no_such_warning(self):
        big = _wildcat()
        big.eu_count = 512
        assert not big.small
        assert "does not transfer" not in big.describe()

    def test_the_threshold_is_explicit(self):
        small, large = _wildcat(), _wildcat()
        small.eu_count = SMALL_DEVICE_EU_THRESHOLD - 1
        large.eu_count = SMALL_DEVICE_EU_THRESHOLD
        assert small.small and not large.small

    def test_an_unknown_eu_count_is_not_treated_as_small(self):
        """Zero means unread, not tiny."""
        unknown = _wildcat()
        unknown.eu_count = 0
        assert not unknown.small


class TestLaunchConstraints:
    def test_the_block_size_bound_is_stated_with_its_number(self):
        rules = " ".join(launch_constraints(_wildcat()))
        assert "1024 work-items" in rules

    def test_the_num_warps_arithmetic_is_spelled_out(self):
        """The exact proposal that failed: num_warps=16 at width 32 is 512 work-items."""
        rules = " ".join(launch_constraints(_wildcat()))
        assert "num_warps=16" in rules
        assert "512 work-items" in rules

    def test_a_narrow_bus_biases_toward_saving_memory_traffic(self):
        rules = " ".join(launch_constraints(_wildcat()))
        assert "64-bit bus" in rules
        assert "bandwidth the binding constraint" in rules

    def test_a_wide_bus_does_not_raise_the_bandwidth_rule(self):
        wide = _wildcat()
        wide.memory_bus_width = 384
        assert not any("binding constraint" in r for r in launch_constraints(wide))


@pytest.mark.xpu
class TestAgainstTheRealDevice:
    def test_the_real_device_reports_facts_not_placeholders(self):
        torch = pytest.importorskip("torch")
        if not (hasattr(torch, "xpu") and torch.xpu.is_available()):
            pytest.skip("no XPU device")
        facts = probe_device()
        assert facts.available
        assert facts.name
        assert facts.max_work_group_size > 0
        assert facts.sub_group_sizes
