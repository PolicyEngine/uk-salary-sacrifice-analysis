"""Salary sacrifice cap reform definitions.

This module implements a salary sacrifice pension contribution cap with
configurable employer and employee behavioral responses.

Scenario Matrix:
---------------
Employer Response:
  - SPREAD_COST: Spread increased NI cost across all employees (cost-neutral)
  - ABSORB_COST: Employer absorbs the full extra NI cost

Employee Response:
  - MAINTAIN_PENSION: Redirect excess to employee pension contributions
  - TAKE_CASH: Don't redirect, take the taxable cash instead

This gives 4 scenarios:
1. Spread + Maintain: Broad haircut, full pension maintained
2. Spread + Take Cash: Broad haircut, lower pension saving
3. Absorb + Maintain: No haircut, full pension maintained (best for employee pension)
4. Absorb + Take Cash: No haircut, max take-home (best for employee cash)
"""

from dataclasses import dataclass
from enum import Enum
from typing import Callable

import numpy as np
from policyengine_core.model_api import Reform
from policyengine_uk.variables.household.income.household_net_income import (
    household_net_income as original_household_net_income,
)


def apply_income_including_pensions(sim) -> None:
    """Apply a structural change that includes pension contributions in net income.

    This modifies household_net_income to add pension contributions, ensuring
    that switching from pension contributions to cash doesn't artificially
    appear as an income gain in distributional analysis.

    Should be called on both baseline and reform simulations.

    Args:
        sim: PolicyEngine Microsimulation object
    """
    from policyengine_core.periods import YEAR
    from policyengine_core.variables import Variable

    # Create the household-level pension variable
    class household_pension_contributions(Variable):
        value_type = float
        entity = sim.tax_benefit_system.entities_by_singular()["household"]
        definition_period = YEAR
        label = "Total pension contributions for the household"
        adds = ["pension_contributions"]

    # Add the new variable to the tax benefit system
    sim.tax_benefit_system.add_variable(household_pension_contributions)

    # Modify household_net_income to include pensions
    original_var = sim.tax_benefit_system.variables["household_net_income"]
    if "household_pension_contributions" not in original_var.adds:
        original_var.adds = original_var.adds + ["household_pension_contributions"]


class EmployerResponse(Enum):
    """How employers respond to increased NI costs."""

    SPREAD_COST = "spread_cost"  # Spread cost across all employees
    ABSORB_COST = "absorb_cost"  # Employer absorbs the full cost
    TARGETED_HAIRCUT = "targeted_haircut"  # Original model: haircut on affected only


class EmployeeResponse(Enum):
    """How employees respond to excess above cap."""

    MAINTAIN_PENSION = "maintain_pension"  # Redirect excess to employee pension
    TAKE_CASH = "take_cash"  # Take taxable cash instead
    PARTIAL_REDIRECT = "partial_redirect"  # Redirect some fraction to pension


@dataclass
class SalarySacrificeCapScenario:
    """Configuration for a salary sacrifice cap scenario.

    Attributes:
        cap_amount: Maximum annual salary sacrifice contribution (GBP)
        employer_ni_rate: Employer NI rate on earnings above threshold
        employer_response: How employer handles increased NI costs
        employee_response: How employee handles excess above cap
        pension_redirect_rate: Fraction of excess redirected to pension
                              (only used with PARTIAL_REDIRECT)
        targeted_haircut_rate: Fraction retained by employer
                              (only used with TARGETED_HAIRCUT)
    """

    cap_amount: float = 2000
    employer_ni_rate: float = 0.138
    employer_response: EmployerResponse = EmployerResponse.SPREAD_COST
    employee_response: EmployeeResponse = EmployeeResponse.MAINTAIN_PENSION
    pension_redirect_rate: float = 1.0  # For PARTIAL_REDIRECT
    targeted_haircut_rate: float = 0.13  # For TARGETED_HAIRCUT

    @property
    def name(self) -> str:
        """Generate a descriptive name for this scenario."""
        parts = [f"cap_{self.cap_amount}"]
        parts.append(self.employer_response.value)
        parts.append(self.employee_response.value)
        if self.employee_response == EmployeeResponse.PARTIAL_REDIRECT:
            parts.append(f"redirect_{int(self.pension_redirect_rate * 100)}pct")
        return "_".join(parts)


# Pre-defined scenarios for the 2x2 matrix
SCENARIOS = {
    "spread_maintain": SalarySacrificeCapScenario(
        employer_response=EmployerResponse.SPREAD_COST,
        employee_response=EmployeeResponse.MAINTAIN_PENSION,
    ),
    "spread_cash": SalarySacrificeCapScenario(
        employer_response=EmployerResponse.SPREAD_COST,
        employee_response=EmployeeResponse.TAKE_CASH,
    ),
    "absorb_maintain": SalarySacrificeCapScenario(
        employer_response=EmployerResponse.ABSORB_COST,
        employee_response=EmployeeResponse.MAINTAIN_PENSION,
    ),
    "absorb_cash": SalarySacrificeCapScenario(
        employer_response=EmployerResponse.ABSORB_COST,
        employee_response=EmployeeResponse.TAKE_CASH,
    ),
    # Original model for comparison
    "targeted_haircut": SalarySacrificeCapScenario(
        employer_response=EmployerResponse.TARGETED_HAIRCUT,
        employee_response=EmployeeResponse.MAINTAIN_PENSION,
    ),
}


def create_salary_sacrifice_cap_reform(
    scenario: SalarySacrificeCapScenario,
    years: list[int] | None = None,
) -> Callable:
    """Create a simulation modifier for the salary sacrifice cap reform.

    Args:
        scenario: Configuration for the reform
        years: Years to apply reform. Defaults to 2026-2030.

    Returns:
        A simulation modifier function for use with PolicyEngine's Microsimulation.
    """
    if years is None:
        years = list(range(2026, 2031))

    def modify(sim):
        for year in years:
            _apply_reform_for_year(sim, year, scenario)

    return modify


def _apply_reform_for_year(sim, year: int, scenario: SalarySacrificeCapScenario):
    """Apply the salary sacrifice cap reform for a single year.

    Args:
        sim: PolicyEngine Microsimulation object
        year: Tax year to modify
        scenario: Reform configuration
    """
    # Get current values
    ss_contrib = sim.calculate(
        "pension_contributions_via_salary_sacrifice", period=year
    ).values
    emp_income = sim.calculate("employment_income", period=year).values
    employee_pension = sim.calculate(
        "employee_pension_contributions", period=year
    ).values

    # Calculate excess above cap
    excess = np.maximum(ss_contrib - scenario.cap_amount, 0)

    # Determine pension redirect amount based on employee response
    if scenario.employee_response == EmployeeResponse.MAINTAIN_PENSION:
        pension_redirect = excess
    elif scenario.employee_response == EmployeeResponse.TAKE_CASH:
        pension_redirect = np.zeros_like(excess)
    elif scenario.employee_response == EmployeeResponse.PARTIAL_REDIRECT:
        pension_redirect = excess * scenario.pension_redirect_rate
    else:
        raise ValueError(f"Unknown employee response: {scenario.employee_response}")

    # Cash portion (what becomes taxable employment income before any haircut)
    cash_portion = excess  # Full excess becomes taxable

    # Apply employer response
    if scenario.employer_response == EmployerResponse.SPREAD_COST:
        # Spread employer NI cost increase across all employees
        total_employer_ni_increase = (excess * scenario.employer_ni_rate).sum()
        total_employment_income = emp_income.sum()

        if total_employment_income > 0:
            broad_haircut_rate = total_employer_ni_increase / total_employment_income
        else:
            broad_haircut_rate = 0

        # All employees get reduced income, then affected get excess added
        new_employment_income = emp_income * (1 - broad_haircut_rate) + cash_portion

    elif scenario.employer_response == EmployerResponse.ABSORB_COST:
        # Employer absorbs full cost - no haircut at all
        new_employment_income = emp_income + cash_portion

    elif scenario.employer_response == EmployerResponse.TARGETED_HAIRCUT:
        # Original model: haircut applied only to affected employees' excess
        haircut = excess * scenario.targeted_haircut_rate
        cash_portion = excess - haircut
        pension_redirect = cash_portion  # In original model, redirect = net excess
        new_employment_income = emp_income + cash_portion

    else:
        raise ValueError(f"Unknown employer response: {scenario.employer_response}")

    # Update employee pension contributions
    new_employee_pension = employee_pension + pension_redirect

    # Cap salary sacrifice at the limit
    new_ss_contrib = np.minimum(ss_contrib, scenario.cap_amount)

    # Apply reformed values
    sim.set_input("employment_income", year, new_employment_income)
    sim.set_input("employee_pension_contributions", year, new_employee_pension)
    sim.set_input("pension_contributions_via_salary_sacrifice", year, new_ss_contrib)


def calculate_affected_population(sim, year: int, cap_amount: float) -> dict:
    """Calculate statistics about the population affected by the cap.

    Args:
        sim: PolicyEngine Microsimulation (baseline)
        year: Tax year
        cap_amount: Cap amount in GBP

    Returns:
        Dictionary with population statistics
    """
    ss_contrib = sim.calculate(
        "pension_contributions_via_salary_sacrifice", period=year
    ).values
    weights = sim.calculate("person_weight", period=year).values
    emp_income = sim.calculate("employment_income", period=year).values

    # Workers with any salary sacrifice
    has_ss = ss_contrib > 0
    ss_contributors = weights[has_ss].sum()

    # Workers exceeding cap
    above_cap = ss_contrib > cap_amount
    affected_workers = weights[above_cap].sum()

    # Total excess
    excess = np.maximum(ss_contrib - cap_amount, 0)
    total_excess = (excess * weights).sum()

    # Average contributions
    avg_ss_all = (
        (ss_contrib[has_ss] * weights[has_ss]).sum() / ss_contributors
        if ss_contributors > 0
        else 0
    )
    avg_ss_above_cap = (
        (ss_contrib[above_cap] * weights[above_cap]).sum() / affected_workers
        if affected_workers > 0
        else 0
    )

    # Total workers
    has_employment = emp_income > 0
    total_workers = weights[has_employment].sum()

    return {
        "total_workers": total_workers,
        "ss_contributors": ss_contributors,
        "ss_contributors_pct": 100 * ss_contributors / total_workers,
        "affected_workers": affected_workers,
        "affected_workers_pct_of_ss": (
            100 * affected_workers / ss_contributors if ss_contributors > 0 else 0
        ),
        "affected_workers_pct_of_all": 100 * affected_workers / total_workers,
        "avg_contribution_all_ss": avg_ss_all,
        "avg_contribution_above_cap": avg_ss_above_cap,
        "total_excess_bn": total_excess / 1e9,
    }


def get_scenario_matrix(
    cap_amounts: list[float] | None = None,
) -> list[SalarySacrificeCapScenario]:
    """Generate all scenarios in the 2x2 matrix for given cap amounts.

    Args:
        cap_amounts: List of cap amounts to test. Defaults to [2000].

    Returns:
        List of scenario configurations
    """
    if cap_amounts is None:
        cap_amounts = [2000]

    scenarios = []
    for cap in cap_amounts:
        for employer in [EmployerResponse.SPREAD_COST, EmployerResponse.ABSORB_COST]:
            for employee in [
                EmployeeResponse.MAINTAIN_PENSION,
                EmployeeResponse.TAKE_CASH,
            ]:
                scenarios.append(
                    SalarySacrificeCapScenario(
                        cap_amount=cap,
                        employer_response=employer,
                        employee_response=employee,
                    )
                )

    return scenarios
