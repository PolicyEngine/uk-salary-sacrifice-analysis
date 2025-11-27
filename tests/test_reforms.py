"""Tests for salary sacrifice cap reform logic.

These tests verify the reform behaves correctly using mock data,
without requiring the full PolicyEngine microsimulation.
"""

from unittest.mock import MagicMock

import numpy as np
import pytest

from salary_sacrifice.reforms import (
    EmployeeResponse,
    EmployerResponse,
    SalarySacrificeCapScenario,
    _apply_reform_for_year,
    calculate_affected_population,
)


class MockSimulation:
    """Mock PolicyEngine simulation for testing."""

    def __init__(self, data: dict):
        """Initialize with test data.

        Args:
            data: Dict mapping variable names to numpy arrays
        """
        self._data = {k: v.copy() for k, v in data.items()}
        self._inputs = {}

    def calculate(self, variable: str, period: int, map_to: str = None):
        """Return mock calculation result."""
        result = MagicMock()
        result.values = self._data.get(variable, np.array([]))
        return result

    def set_input(self, variable: str, period: int, values: np.ndarray):
        """Record input being set."""
        self._inputs[variable] = values.copy()

    def get_input(self, variable: str) -> np.ndarray:
        """Get the value that was set."""
        return self._inputs.get(variable)


class TestSalarySacrificeCapScenario:
    """Tests for scenario configuration."""

    def test_default_values(self):
        scenario = SalarySacrificeCapScenario()
        assert scenario.cap_amount == 2000
        assert scenario.employer_ni_rate == 0.138
        assert scenario.employer_response == EmployerResponse.SPREAD_COST
        assert scenario.employee_response == EmployeeResponse.MAINTAIN_PENSION

    def test_custom_values(self):
        scenario = SalarySacrificeCapScenario(
            cap_amount=3000,
            employer_ni_rate=0.15,
            employer_response=EmployerResponse.ABSORB_COST,
            employee_response=EmployeeResponse.TAKE_CASH,
        )
        assert scenario.cap_amount == 3000
        assert scenario.employer_ni_rate == 0.15
        assert scenario.employer_response == EmployerResponse.ABSORB_COST
        assert scenario.employee_response == EmployeeResponse.TAKE_CASH


class TestSpreadCostMaintainPension:
    """Tests for SPREAD_COST + MAINTAIN_PENSION scenario (default)."""

    def test_no_change_when_under_cap(self):
        """Workers under cap should only see broad haircut, no conversion."""
        sim = MockSimulation(
            {
                "pension_contributions_via_salary_sacrifice": np.array(
                    [1000, 1500, 500]
                ),
                "employment_income": np.array([50000, 60000, 40000]),
                "employee_pension_contributions": np.array([0, 0, 0]),
            }
        )

        scenario = SalarySacrificeCapScenario(cap_amount=2000)
        _apply_reform_for_year(sim, 2026, scenario)

        # No excess, so no employer NI increase, so no haircut
        # Employment income should be unchanged (plus 0 excess)
        np.testing.assert_array_almost_equal(
            sim.get_input("employment_income"),
            np.array([50000, 60000, 40000]),
        )

        # Salary sacrifice unchanged (all under cap)
        np.testing.assert_array_almost_equal(
            sim.get_input("pension_contributions_via_salary_sacrifice"),
            np.array([1000, 1500, 500]),
        )

        # No change to employee pension contributions
        np.testing.assert_array_almost_equal(
            sim.get_input("employee_pension_contributions"),
            np.array([0, 0, 0]),
        )

    def test_excess_redirected_to_employee_pension(self):
        """Excess above cap should go to employee pension contributions."""
        sim = MockSimulation(
            {
                "pension_contributions_via_salary_sacrifice": np.array([5000, 1000]),
                "employment_income": np.array([50000, 50000]),
                "employee_pension_contributions": np.array([0, 0]),
            }
        )

        scenario = SalarySacrificeCapScenario(cap_amount=2000)
        _apply_reform_for_year(sim, 2026, scenario)

        # First worker has £3000 excess, second has none
        employee_pension = sim.get_input("employee_pension_contributions")

        # Full excess goes to employee pension (MAINTAIN_PENSION response)
        assert employee_pension[0] == 3000
        assert employee_pension[1] == 0

    def test_salary_sacrifice_capped(self):
        """Salary sacrifice should be capped at the limit."""
        sim = MockSimulation(
            {
                "pension_contributions_via_salary_sacrifice": np.array(
                    [5000, 8000, 1000]
                ),
                "employment_income": np.array([50000, 60000, 40000]),
                "employee_pension_contributions": np.array([0, 0, 0]),
            }
        )

        scenario = SalarySacrificeCapScenario(cap_amount=2000)
        _apply_reform_for_year(sim, 2026, scenario)

        ss = sim.get_input("pension_contributions_via_salary_sacrifice")
        np.testing.assert_array_almost_equal(ss, np.array([2000, 2000, 1000]))

    def test_total_pension_maintained(self):
        """Total pension contribution should be maintained for affected workers."""
        original_ss = np.array([5000, 3000, 1000])
        original_employee = np.array([1000, 500, 200])

        sim = MockSimulation(
            {
                "pension_contributions_via_salary_sacrifice": original_ss,
                "employment_income": np.array([50000, 60000, 40000]),
                "employee_pension_contributions": original_employee,
            }
        )

        scenario = SalarySacrificeCapScenario(cap_amount=2000)
        _apply_reform_for_year(sim, 2026, scenario)

        new_ss = sim.get_input("pension_contributions_via_salary_sacrifice")
        new_employee = sim.get_input("employee_pension_contributions")

        # Total pension = new_ss + new_employee equals original_ss + original_employee
        original_total = original_ss + original_employee
        new_total = new_ss + new_employee

        np.testing.assert_array_almost_equal(original_total, new_total)

    def test_employer_cost_spread_broadly(self):
        """Employer NI cost increase should be spread across all workers."""
        sim = MockSimulation(
            {
                "pension_contributions_via_salary_sacrifice": np.array([5000, 0]),
                "employment_income": np.array([50000, 50000]),  # Equal income
                "employee_pension_contributions": np.array([0, 0]),
            }
        )

        scenario = SalarySacrificeCapScenario(
            cap_amount=2000,
            employer_ni_rate=0.138,
        )
        _apply_reform_for_year(sim, 2026, scenario)

        emp_income = sim.get_input("employment_income")

        # Excess = 3000, employer NI increase = 3000 * 0.138 = 414
        # Total employment income = 100000
        # Haircut rate = 414 / 100000 = 0.00414
        # Worker 1: 50000 * (1 - 0.00414) + 3000 = 52793
        # Worker 2: 50000 * (1 - 0.00414) + 0 = 49793

        expected_haircut = 3000 * 0.138 / 100000
        expected_w1 = 50000 * (1 - expected_haircut) + 3000
        expected_w2 = 50000 * (1 - expected_haircut)

        np.testing.assert_array_almost_equal(
            emp_income, np.array([expected_w1, expected_w2])
        )

    def test_total_employer_cost_constant(self):
        """Total employer cost should remain approximately constant."""
        original_income = np.array([50000, 60000, 40000])
        original_ss = np.array([5000, 3000, 1000])

        sim = MockSimulation(
            {
                "pension_contributions_via_salary_sacrifice": original_ss,
                "employment_income": original_income,
                "employee_pension_contributions": np.array([0, 0, 0]),
            }
        )

        scenario = SalarySacrificeCapScenario(
            cap_amount=2000,
            employer_ni_rate=0.138,
        )

        excess = np.maximum(original_ss - 2000, 0)
        total_excess = excess.sum()

        _apply_reform_for_year(sim, 2026, scenario)

        new_income = sim.get_input("employment_income")

        # The reduction in total income should equal the employer NI cost increase
        income_reduction = original_income.sum() - (new_income.sum() - total_excess)
        employer_ni_increase = total_excess * 0.138

        np.testing.assert_almost_equal(
            income_reduction, employer_ni_increase, decimal=0
        )


class TestSpreadCostTakeCash:
    """Tests for SPREAD_COST + TAKE_CASH scenario."""

    def test_no_pension_redirect(self):
        """Excess should not be redirected to pension when TAKE_CASH."""
        sim = MockSimulation(
            {
                "pension_contributions_via_salary_sacrifice": np.array([5000, 1000]),
                "employment_income": np.array([50000, 50000]),
                "employee_pension_contributions": np.array([0, 0]),
            }
        )

        scenario = SalarySacrificeCapScenario(
            cap_amount=2000,
            employer_response=EmployerResponse.SPREAD_COST,
            employee_response=EmployeeResponse.TAKE_CASH,
        )
        _apply_reform_for_year(sim, 2026, scenario)

        # Employee pension should stay at 0 (no redirect)
        employee_pension = sim.get_input("employee_pension_contributions")
        np.testing.assert_array_almost_equal(employee_pension, np.array([0, 0]))

        # But salary sacrifice should still be capped
        ss = sim.get_input("pension_contributions_via_salary_sacrifice")
        np.testing.assert_array_almost_equal(ss, np.array([2000, 1000]))


class TestAbsorbCostMaintainPension:
    """Tests for ABSORB_COST + MAINTAIN_PENSION scenario."""

    def test_no_broad_haircut(self):
        """Employer absorbs cost, so no haircut applied."""
        sim = MockSimulation(
            {
                "pension_contributions_via_salary_sacrifice": np.array([5000, 0]),
                "employment_income": np.array([50000, 50000]),
                "employee_pension_contributions": np.array([0, 0]),
            }
        )

        scenario = SalarySacrificeCapScenario(
            cap_amount=2000,
            employer_response=EmployerResponse.ABSORB_COST,
            employee_response=EmployeeResponse.MAINTAIN_PENSION,
        )
        _apply_reform_for_year(sim, 2026, scenario)

        emp_income = sim.get_input("employment_income")

        # Worker 1: 50000 + 3000 (excess becomes taxable income)
        # Worker 2: 50000 (no change, no haircut applied)
        np.testing.assert_array_almost_equal(emp_income, np.array([53000, 50000]))

    def test_pension_still_redirected(self):
        """Excess should still go to employee pension with MAINTAIN_PENSION."""
        sim = MockSimulation(
            {
                "pension_contributions_via_salary_sacrifice": np.array([5000]),
                "employment_income": np.array([50000]),
                "employee_pension_contributions": np.array([0]),
            }
        )

        scenario = SalarySacrificeCapScenario(
            cap_amount=2000,
            employer_response=EmployerResponse.ABSORB_COST,
            employee_response=EmployeeResponse.MAINTAIN_PENSION,
        )
        _apply_reform_for_year(sim, 2026, scenario)

        employee_pension = sim.get_input("employee_pension_contributions")
        assert employee_pension[0] == 3000


class TestAbsorbCostTakeCash:
    """Tests for ABSORB_COST + TAKE_CASH scenario (best for employee cash)."""

    def test_maximum_take_home(self):
        """Maximizes take-home: no haircut, no pension redirect."""
        sim = MockSimulation(
            {
                "pension_contributions_via_salary_sacrifice": np.array([5000]),
                "employment_income": np.array([50000]),
                "employee_pension_contributions": np.array([0]),
            }
        )

        scenario = SalarySacrificeCapScenario(
            cap_amount=2000,
            employer_response=EmployerResponse.ABSORB_COST,
            employee_response=EmployeeResponse.TAKE_CASH,
        )
        _apply_reform_for_year(sim, 2026, scenario)

        # Full excess becomes taxable income (no haircut)
        emp_income = sim.get_input("employment_income")
        assert emp_income[0] == 53000

        # No redirect to pension
        employee_pension = sim.get_input("employee_pension_contributions")
        assert employee_pension[0] == 0


class TestTargetedHaircutReform:
    """Tests for the targeted haircut approach (original model, for comparison)."""

    def test_haircut_applied_to_excess(self):
        """13% haircut should be applied to excess only."""
        sim = MockSimulation(
            {
                "pension_contributions_via_salary_sacrifice": np.array([5000]),
                "employment_income": np.array([50000]),
                "employee_pension_contributions": np.array([0]),
            }
        )

        scenario = SalarySacrificeCapScenario(
            cap_amount=2000,
            employer_response=EmployerResponse.TARGETED_HAIRCUT,
            targeted_haircut_rate=0.13,
        )
        _apply_reform_for_year(sim, 2026, scenario)

        # Excess = 3000, net excess = 3000 * 0.87 = 2610
        employee_pension = sim.get_input("employee_pension_contributions")
        assert employee_pension[0] == pytest.approx(2610)

        emp_income = sim.get_input("employment_income")
        assert emp_income[0] == pytest.approx(50000 + 2610)

    def test_unaffected_workers_unchanged(self):
        """Workers under cap should be unchanged in targeted model."""
        sim = MockSimulation(
            {
                "pension_contributions_via_salary_sacrifice": np.array([1000]),
                "employment_income": np.array([50000]),
                "employee_pension_contributions": np.array([500]),
            }
        )

        scenario = SalarySacrificeCapScenario(
            cap_amount=2000,
            employer_response=EmployerResponse.TARGETED_HAIRCUT,
        )
        _apply_reform_for_year(sim, 2026, scenario)

        # No excess, no change
        assert sim.get_input("employment_income")[0] == 50000
        assert sim.get_input("employee_pension_contributions")[0] == 500


class TestCalculateAffectedPopulation:
    """Tests for population statistics calculation."""

    def test_basic_statistics(self):
        """Test calculation of affected population stats."""
        sim = MockSimulation(
            {
                "pension_contributions_via_salary_sacrifice": np.array(
                    [5000, 3000, 1000, 0, 0]
                ),
                "employment_income": np.array([50000, 60000, 40000, 30000, 0]),
                "person_weight": np.array([1000, 1000, 1000, 1000, 1000]),
            }
        )

        stats = calculate_affected_population(sim, 2026, cap_amount=2000)

        # 4 workers (last person has 0 employment income)
        assert stats["total_workers"] == 4000

        # 3 have salary sacrifice
        assert stats["ss_contributors"] == 3000

        # 2 above cap (5000 and 3000)
        assert stats["affected_workers"] == 2000

        # Total excess = (5000-2000)*1000 + (3000-2000)*1000 = 4,000,000
        assert stats["total_excess_bn"] == pytest.approx(4_000_000 / 1e9)


@pytest.mark.parametrize(
    "cap,expected_excess",
    [
        (2000, 3000),  # 5000 - 2000
        (3000, 2000),  # 5000 - 3000
        (5000, 0),  # At cap, no excess
        (6000, 0),  # Above contribution, no excess
    ],
)
def test_excess_calculation_various_caps(cap, expected_excess):
    """Test excess calculation with various cap levels."""
    sim = MockSimulation(
        {
            "pension_contributions_via_salary_sacrifice": np.array([5000]),
            "employment_income": np.array([50000]),
            "employee_pension_contributions": np.array([0]),
        }
    )

    scenario = SalarySacrificeCapScenario(cap_amount=cap)
    _apply_reform_for_year(sim, 2026, scenario)

    employee_pension = sim.get_input("employee_pension_contributions")
    assert employee_pension[0] == pytest.approx(expected_excess)
