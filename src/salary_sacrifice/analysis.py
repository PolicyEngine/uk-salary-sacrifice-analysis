"""Analysis functions for running salary sacrifice cap simulations."""

import pandas as pd

from salary_sacrifice.reforms import (
    SalarySacrificeCapScenario,
    calculate_affected_population,
    create_salary_sacrifice_cap_reform,
    get_scenario_matrix,
)


def run_scenario(
    scenario: SalarySacrificeCapScenario,
    years: list[int] | None = None,
    dataset_path: str | None = None,
) -> dict:
    """Run a single scenario and return results.

    Args:
        scenario: Scenario configuration
        years: Years to analyze. Defaults to [2026, 2027, 2028, 2029].
        dataset_path: Optional path to dataset file.

    Returns:
        Dictionary with revenue and distributional results.
    """
    from policyengine_uk import Microsimulation
    from policyengine_uk.utils.scenario import Scenario

    if years is None:
        years = [2026, 2027, 2028, 2029]

    # Create simulations
    kwargs = {}
    if dataset_path:
        from policyengine_uk.data import UKSingleYearDataset

        kwargs["dataset"] = UKSingleYearDataset(file_path=dataset_path)

    baseline = Microsimulation(**kwargs)

    reform_modifier = create_salary_sacrifice_cap_reform(scenario, years=years)
    pe_scenario = Scenario(simulation_modifier=reform_modifier)
    reformed = Microsimulation(scenario=pe_scenario, **kwargs)

    results = {
        "scenario": scenario.name,
        "cap_amount": scenario.cap_amount,
        "employer_response": scenario.employer_response.value,
        "employee_response": scenario.employee_response.value,
        "revenue_by_year": {},
        "affected_population": {},
    }

    for year in years:
        # Revenue impact
        baseline_balance = baseline.calculate("gov_balance", period=year).sum()
        reformed_balance = reformed.calculate("gov_balance", period=year).sum()
        revenue_change = (reformed_balance - baseline_balance) / 1e9

        results["revenue_by_year"][year] = revenue_change

        # Population stats (only need once, use first year)
        if year == years[0]:
            results["affected_population"] = calculate_affected_population(
                baseline, year, scenario.cap_amount
            )

    results["total_revenue_bn"] = sum(results["revenue_by_year"].values())
    results["avg_annual_revenue_bn"] = results["total_revenue_bn"] / len(years)

    return results


def run_scenario_matrix(
    cap_amounts: list[float] | None = None,
    years: list[int] | None = None,
    dataset_path: str | None = None,
    include_targeted_haircut: bool = True,
) -> pd.DataFrame:
    """Run all scenarios in the 2x2 matrix and return results as DataFrame.

    Args:
        cap_amounts: List of cap amounts to test.
        years: Years to analyze.
        dataset_path: Optional path to dataset file.
        include_targeted_haircut: Whether to include original targeted haircut model.

    Returns:
        DataFrame with results for all scenarios.
    """
    from salary_sacrifice.reforms import (
        EmployeeResponse,
        EmployerResponse,
    )

    scenarios = get_scenario_matrix(cap_amounts)

    # Optionally add targeted haircut for comparison
    if include_targeted_haircut:
        for cap in cap_amounts or [2000]:
            scenarios.append(
                SalarySacrificeCapScenario(
                    cap_amount=cap,
                    employer_response=EmployerResponse.TARGETED_HAIRCUT,
                    employee_response=EmployeeResponse.MAINTAIN_PENSION,
                )
            )

    all_results = []
    for scenario in scenarios:
        print(f"Running scenario: {scenario.name}")
        results = run_scenario(scenario, years=years, dataset_path=dataset_path)
        all_results.append(results)

    # Convert to DataFrame
    rows = []
    for r in all_results:
        row = {
            "scenario": r["scenario"],
            "cap_amount": r["cap_amount"],
            "employer_response": r["employer_response"],
            "employee_response": r["employee_response"],
            "avg_annual_revenue_bn": r["avg_annual_revenue_bn"],
            "affected_workers": r["affected_population"]["affected_workers"],
            "total_excess_bn": r["affected_population"]["total_excess_bn"],
        }
        # Add yearly revenue
        for year, rev in r["revenue_by_year"].items():
            row[f"revenue_{year}_bn"] = rev
        rows.append(row)

    return pd.DataFrame(rows)


def calculate_distributional_impact(
    scenario: SalarySacrificeCapScenario,
    year: int = 2026,
    dataset_path: str | None = None,
) -> pd.DataFrame:
    """Calculate distributional impact by income decile.

    Args:
        scenario: Scenario configuration
        year: Year to analyze
        dataset_path: Optional path to dataset file

    Returns:
        DataFrame with impact by income decile
    """
    from policyengine_uk import Microsimulation
    from policyengine_uk.utils.scenario import Scenario

    kwargs = {}
    if dataset_path:
        from policyengine_uk.data import UKSingleYearDataset

        kwargs["dataset"] = UKSingleYearDataset(file_path=dataset_path)

    baseline = Microsimulation(**kwargs)
    reform_modifier = create_salary_sacrifice_cap_reform(scenario, years=[year])
    pe_scenario = Scenario(simulation_modifier=reform_modifier)
    reformed = Microsimulation(scenario=pe_scenario, **kwargs)

    # Get data at household level - use PolicyEngine series with built-in weights
    baseline_income = baseline.calculate("household_net_income", period=year)
    reformed_income = reformed.calculate("household_net_income", period=year)
    deciles = baseline.calculate("household_income_decile", period=year)
    weights = baseline.calculate("household_weight", period=year)

    results = []
    for decile in range(1, 11):
        mask = deciles == decile

        # Use .sum() which respects weights in PolicyEngine series
        baseline_sum = baseline_income[mask].sum()
        reformed_sum = reformed_income[mask].sum()
        total_weight = weights[mask].sum()

        if total_weight == 0:
            continue

        avg_baseline = baseline_sum / total_weight
        avg_reformed = reformed_sum / total_weight
        avg_change = avg_reformed - avg_baseline
        pct_change = 100 * avg_change / avg_baseline if avg_baseline != 0 else 0

        results.append(
            {
                "decile": decile,
                "avg_baseline_income": avg_baseline,
                "avg_reformed_income": avg_reformed,
                "avg_change": avg_change,
                "pct_change": pct_change,
                "population": total_weight,
            }
        )

    return pd.DataFrame(results)
