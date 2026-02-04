"""Batch data generation for the salary sacrifice cap analysis app.

Iterates over cap levels (£0-£10k), scenarios (4), and years (2)
to produce a single JSON file consumed by the React frontend.
"""

import json
import time
from pathlib import Path

import click

from salary_sacrifice.analysis import calculate_distributional_impact, run_scenario
from salary_sacrifice.reforms import (
    EmployeeResponse,
    EmployerResponse,
    SalarySacrificeCapScenario,
)

# Configuration
CAP_LEVELS = list(range(0, 7_000, 1_000))  # £0 to £6k in £1k steps
SCENARIOS = [
    (
        "spread_maintain",
        EmployerResponse.SPREAD_COST,
        EmployeeResponse.MAINTAIN_PENSION,
    ),
    ("spread_cash", EmployerResponse.SPREAD_COST, EmployeeResponse.TAKE_CASH),
    (
        "absorb_maintain",
        EmployerResponse.ABSORB_COST,
        EmployeeResponse.MAINTAIN_PENSION,
    ),
    ("absorb_cash", EmployerResponse.ABSORB_COST, EmployeeResponse.TAKE_CASH),
]
YEARS = [2029, 2030]

OUTPUT_DIR = Path(__file__).resolve().parents[2] / "app" / "public" / "data"
OUTPUT_FILE = OUTPUT_DIR / "simulation-results.json"
PROGRESS_FILE = OUTPUT_DIR / "progress.json"


def load_progress() -> dict:
    """Load previously completed simulation combos."""
    if PROGRESS_FILE.exists():
        return json.loads(PROGRESS_FILE.read_text())
    return {"completed": {}, "results": {}}


def save_progress(progress: dict):
    """Save progress to disk for resumability."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PROGRESS_FILE.write_text(json.dumps(progress, indent=2))


def combo_key(cap: int, year: int, scenario_name: str) -> str:
    return f"{cap}_{year}_{scenario_name}"


def run_single_combo(cap, year, scenario_name, employer_response, employee_response):
    """Run a single cap/year/scenario combination."""
    scenario = SalarySacrificeCapScenario(
        cap_amount=cap,
        employer_response=employer_response,
        employee_response=employee_response,
    )

    # Revenue and affected population
    result = run_scenario(scenario, years=[year])

    # Distributional impact
    dist_df = calculate_distributional_impact(scenario, year=year)

    return {
        "revenue_bn": round(float(result["revenue_by_year"][year]), 4),
        "affected_workers": int(result["affected_population"]["affected_workers"]),
        "total_excess_bn": round(
            float(result["affected_population"]["total_excess_bn"]), 4
        ),
        "distributional": {
            "deciles": [int(v) for v in dist_df["decile"].tolist()],
            "pct_change": [round(float(v), 4) for v in dist_df["pct_change"].tolist()],
            "abs_change": [round(float(v), 2) for v in dist_df["avg_change"].tolist()],
            "share_affected": [
                round(float(v), 6) for v in dist_df["share_affected"].tolist()
            ],
        },
    }


@click.command("generate-data")
@click.option("--resume/--no-resume", default=True, help="Resume from previous run")
@click.option(
    "--output",
    "-o",
    default=None,
    type=click.Path(),
    help="Output JSON path (default: app/public/data/simulation-results.json)",
)
def generate_data(resume, output):
    """Generate simulation results JSON for all cap/year/scenario combos."""
    output_path = Path(output) if output else OUTPUT_FILE

    total_combos = len(CAP_LEVELS) * len(YEARS) * len(SCENARIOS)

    progress = load_progress() if resume else {"completed": {}, "results": {}}
    completed_count = len(progress["completed"])

    click.echo(f"Total combos: {total_combos}")
    click.echo(f"Already completed: {completed_count}")
    click.echo(f"Remaining: {total_combos - completed_count}")
    click.echo()

    for cap in CAP_LEVELS:
        cap_key = str(cap)
        if cap_key not in progress["results"]:
            progress["results"][cap_key] = {}

        for year in YEARS:
            year_key = str(year)
            if year_key not in progress["results"][cap_key]:
                progress["results"][cap_key][year_key] = {}

            for scenario_name, employer_resp, employee_resp in SCENARIOS:
                key = combo_key(cap, year, scenario_name)
                if key in progress["completed"]:
                    continue

                click.echo(
                    f"[{completed_count + 1}/{total_combos}] "
                    f"cap=£{cap:,} year={year} scenario={scenario_name}..."
                )
                start = time.time()

                try:
                    result = run_single_combo(
                        cap, year, scenario_name, employer_resp, employee_resp
                    )
                    progress["results"][cap_key][year_key][scenario_name] = result
                    progress["completed"][key] = True
                    completed_count += 1

                    elapsed = time.time() - start
                    click.echo(
                        f"  Done in {elapsed:.1f}s "
                        f"(revenue=£{result['revenue_bn']:.2f}bn)"
                    )
                except Exception as e:
                    click.echo(f"  ERROR: {e}", err=True)
                    # Save progress and continue
                    save_progress(progress)
                    continue

                # Save progress after each combo
                save_progress(progress)

    # Write final output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    final = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "caps": CAP_LEVELS,
        "years": YEARS,
        "scenarios": [s[0] for s in SCENARIOS],
        "results": progress["results"],
    }
    output_path.write_text(json.dumps(final, indent=2))
    click.echo(f"\nResults written to: {output_path}")

    # Clean up progress file
    if PROGRESS_FILE.exists():
        PROGRESS_FILE.unlink()
    click.echo("Done!")
