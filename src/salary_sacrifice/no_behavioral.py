"""No-behavioral-response analysis using PolicyEngine's native cap parameter.

This module implements a simpler analysis that directly uses PolicyEngine's
built-in salary sacrifice cap parameter without modeling employer/employee
behavioral responses. It uses proper household weights for population-
representative statistics.

Methodology from: House of Commons Library salary sacrifice analysis
"""

import json
import time
from pathlib import Path

import click
import pandas as pd

# Configuration
CAP_LEVELS = list(range(0, 7_000, 1_000))  # £0 to £6k in £1k steps
YEARS = [2029, 2030]
BASELINES = {
    "none": 1e10,  # No cap (effectively unlimited) - pre-budget
    "2000": 2000,  # £2k cap - current law (Autumn Budget 2025)
}

OUTPUT_DIR = Path(__file__).resolve().parents[2] / "app" / "public" / "data"
OUTPUT_FILE = OUTPUT_DIR / "no-behavioral-results.json"
PROGRESS_FILE = OUTPUT_DIR / "no-behavioral-progress.json"


def create_simulations(year: int, cap_reform: int, baseline_cap: float):
    """Create baseline and reform simulations.

    Args:
        year: The tax year to simulate
        cap_reform: The salary sacrifice cap to apply in reform
        baseline_cap: The baseline cap to compare against

    Returns:
        Tuple of (baseline, reform) Microsimulation objects
    """
    from policyengine_uk import Microsimulation

    # Baseline simulation with specified baseline cap
    baseline = Microsimulation(
        reform={
            "gov.hmrc.national_insurance.salary_sacrifice_pension_cap": {
                f"{year}-01-01": baseline_cap,
            }
        }
    )

    # Reform: apply the specified cap
    reform = Microsimulation(
        reform={
            "gov.hmrc.national_insurance.salary_sacrifice_pension_cap": {
                f"{year}-01-01": cap_reform,
            }
        }
    )
    return baseline, reform


def calculate_budgetary_impact(baseline, reform, year: int) -> float:
    """Calculate change in government revenue (£bn)."""
    baseline_revenue = baseline.calculate("gov_balance", year).sum()
    reform_revenue = reform.calculate("gov_balance", year).sum()
    return (reform_revenue - baseline_revenue) / 1e9


def calculate_distributional_impact_weighted(
    baseline, reform, year: int
) -> pd.DataFrame:
    """Calculate distributional impact by income decile using proper weights.

    Uses household_weight for population-representative averages.
    """
    baseline_income = baseline.calculate("household_net_income", year)
    reform_income = reform.calculate("household_net_income", year)
    decile = baseline.calculate("household_income_decile", year)
    weight = baseline.calculate("household_weight", year)

    change = reform_income - baseline_income

    df = pd.DataFrame(
        {
            "decile": decile.values,
            "change": change.values,
            "baseline_income": baseline_income.values,
            "weight": weight.values,
        }
    )

    results = []
    for d in range(1, 11):
        decile_data = df[df["decile"] == d]
        total_weight = decile_data["weight"].sum()

        if total_weight == 0:
            continue

        avg_change = (
            decile_data["change"] * decile_data["weight"]
        ).sum() / total_weight
        avg_baseline = (
            decile_data["baseline_income"] * decile_data["weight"]
        ).sum() / total_weight
        pct_change = (avg_change / avg_baseline) * 100 if avg_baseline > 0 else 0

        results.append(
            {
                "decile": d,
                "avg_change": avg_change,
                "avg_baseline": avg_baseline,
                "pct_change": pct_change,
            }
        )

    return pd.DataFrame(results)


def calculate_winners_losers(baseline, reform, year: int) -> pd.DataFrame:
    """Calculate winners/losers by income decile (% of people)."""
    import numpy as np

    baseline_income = baseline.calculate("household_net_income", year)
    reform_income = reform.calculate("household_net_income", year)
    household_decile = baseline.calculate("household_income_decile", year)
    household_count_people = baseline.calculate("household_count_people", year)

    income_change = reform_income - baseline_income
    capped_baseline = np.maximum(baseline_income.values, 1)
    pct_change = (income_change.values / capped_baseline) * 100
    deciles = household_decile.values
    valid_mask = deciles >= 1
    threshold = 0.0001

    results = []
    for d in range(1, 11):
        decile_mask = valid_mask & (deciles == d)
        total_people = household_count_people[decile_mask].sum()

        if total_people == 0:
            continue

        win_mask = decile_mask & (pct_change > threshold)
        winners = household_count_people[win_mask].sum()

        lose_mask = decile_mask & (pct_change < -threshold)
        losers = household_count_people[lose_mask].sum()

        no_change_mask = (
            decile_mask & (pct_change >= -threshold) & (pct_change <= threshold)
        )
        no_change = household_count_people[no_change_mask].sum()

        results.append(
            {
                "decile": d,
                "pct_winners": round(100 * winners / total_people, 2),
                "pct_losers": round(100 * losers / total_people, 2),
                "pct_no_change": round(100 * no_change / total_people, 2),
            }
        )

    return pd.DataFrame(results)


def run_single_cap(cap: int, year: int, baseline_cap: float) -> dict:
    """Run analysis for a single cap level, year, and baseline."""
    baseline, reform = create_simulations(year, cap, baseline_cap)

    # Budgetary impact
    revenue_bn = calculate_budgetary_impact(baseline, reform, year)

    # Distributional impact (weighted)
    dist_df = calculate_distributional_impact_weighted(baseline, reform, year)

    # Winners/losers
    wl_df = calculate_winners_losers(baseline, reform, year)

    return {
        "revenue_bn": round(float(revenue_bn), 4),
        "distributional": {
            "deciles": [int(v) for v in dist_df["decile"].tolist()],
            "pct_change": [round(float(v), 4) for v in dist_df["pct_change"].tolist()],
            "abs_change": [round(float(v), 2) for v in dist_df["avg_change"].tolist()],
        },
        "winners_losers": {
            "deciles": [int(v) for v in wl_df["decile"].tolist()],
            "pct_winners": [float(v) for v in wl_df["pct_winners"].tolist()],
            "pct_losers": [float(v) for v in wl_df["pct_losers"].tolist()],
            "pct_no_change": [float(v) for v in wl_df["pct_no_change"].tolist()],
        },
    }


def load_progress() -> dict:
    """Load previously completed simulation combos."""
    if PROGRESS_FILE.exists():
        return json.loads(PROGRESS_FILE.read_text())
    return {"completed": {}, "results": {}}


def save_progress(progress: dict):
    """Save progress to disk for resumability."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PROGRESS_FILE.write_text(json.dumps(progress, indent=2))


@click.command("generate-no-behavioral")
@click.option("--resume/--no-resume", default=True, help="Resume from previous run")
@click.option(
    "--output",
    "-o",
    default=None,
    type=click.Path(),
    help="Output JSON path",
)
def generate_no_behavioral_data(resume, output):
    """Generate no-behavioral-response simulation results for both baselines."""
    output_path = Path(output) if output else OUTPUT_FILE

    total_combos = len(CAP_LEVELS) * len(YEARS) * len(BASELINES)

    progress = load_progress() if resume else {"completed": {}, "results": {}}
    completed_count = len(progress["completed"])

    click.echo("=" * 60)
    click.echo("No Behavioral Response Analysis")
    click.echo("=" * 60)
    click.echo(f"Baselines: {list(BASELINES.keys())}")
    click.echo(f"Total combos: {total_combos}")
    click.echo(f"Already completed: {completed_count}")
    click.echo(f"Remaining: {total_combos - completed_count}")
    click.echo()

    for baseline_key, baseline_cap in BASELINES.items():
        if baseline_key not in progress["results"]:
            progress["results"][baseline_key] = {}

        baseline_label = (
            "no cap" if baseline_key == "none" else f"£{baseline_cap:,} cap"
        )
        click.echo(f"\n--- Baseline: {baseline_label} ---")

        for cap in CAP_LEVELS:
            cap_key = str(cap)
            if cap_key not in progress["results"][baseline_key]:
                progress["results"][baseline_key][cap_key] = {}

            for year in YEARS:
                year_key = str(year)
                combo_key = f"{baseline_key}_{cap}_{year}"

                if combo_key in progress["completed"]:
                    continue

                click.echo(
                    f"[{completed_count + 1}/{total_combos}] "
                    f"baseline={baseline_key} cap=£{cap:,} year={year}..."
                )
                start = time.time()

                try:
                    result = run_single_cap(cap, year, baseline_cap)
                    progress["results"][baseline_key][cap_key][year_key] = result
                    progress["completed"][combo_key] = True
                    completed_count += 1

                    elapsed = time.time() - start
                    click.echo(
                        f"  Done in {elapsed:.1f}s "
                        f"(revenue=£{result['revenue_bn']:.2f}bn)"
                    )
                except Exception as e:
                    click.echo(f"  ERROR: {e}", err=True)
                    save_progress(progress)
                    continue

                save_progress(progress)

    # Write final output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    final = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "caps": CAP_LEVELS,
        "years": YEARS,
        "baselines": list(BASELINES.keys()),
        "methodology": "no_behavioral_response",
        "description": (
            "Uses PolicyEngine's native salary sacrifice cap parameter. "
            "No employer/employee behavioral modeling. "
            "Weighted averages using household_weight. "
            "Results generated for each baseline assumption."
        ),
        "results": progress["results"],
    }
    output_path.write_text(json.dumps(final, indent=2))
    click.echo(f"\nResults written to: {output_path}")

    # Clean up progress file
    if PROGRESS_FILE.exists():
        PROGRESS_FILE.unlink()
    click.echo("Done!")


if __name__ == "__main__":
    generate_no_behavioral_data()
