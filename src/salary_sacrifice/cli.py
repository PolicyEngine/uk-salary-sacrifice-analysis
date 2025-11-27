"""Command-line interface for salary sacrifice cap analysis."""

from pathlib import Path

import click


@click.group()
def main():
    """UK Salary Sacrifice Cap Analysis CLI."""
    pass


@click.command()
@click.option(
    "--cap",
    "-c",
    default=2000,
    type=float,
    help="Cap amount in GBP (default: 2000)",
)
@click.option(
    "--year",
    "-y",
    default=2026,
    type=int,
    help="Year to analyze (default: 2026)",
)
@click.option(
    "--dataset",
    "-d",
    default=None,
    type=click.Path(exists=True),
    help="Path to dataset file",
)
@click.option(
    "--output",
    "-o",
    default="outputs",
    type=click.Path(),
    help="Output directory (default: outputs)",
)
def run_matrix(cap, year, dataset, output):
    """Run the 2x2 scenario matrix."""

    from salary_sacrifice.analysis import run_scenario_matrix

    output_dir = Path(output)
    output_dir.mkdir(exist_ok=True)

    click.echo(f"Running 2x2 scenario matrix for £{cap:,.0f} cap...")
    click.echo()

    results = run_scenario_matrix(
        cap_amounts=[cap],
        years=[year],
        dataset_path=dataset,
        include_targeted_haircut=False,  # Just 2x2
    )

    # Display results
    click.echo("=" * 70)
    click.echo("RESULTS: 2x2 Scenario Matrix")
    click.echo("=" * 70)
    click.echo()

    for _, row in results.iterrows():
        emp_resp = row["employer_response"]
        ee_resp = row["employee_response"]
        click.echo(f"Employer: {emp_resp:15} | Employee: {ee_resp:15}")
        click.echo(f"  Revenue: £{row['avg_annual_revenue_bn']:.2f}bn")
        click.echo()

    # Save to CSV
    output_file = output_dir / f"scenario_matrix_cap_{int(cap)}_{year}.csv"
    results.to_csv(output_file, index=False)
    click.echo(f"Results saved to: {output_file}")


@click.command()
@click.option(
    "--cap",
    "-c",
    default=2000,
    type=float,
    help="Cap amount in GBP",
)
@click.option(
    "--employer",
    "-e",
    type=click.Choice(["spread", "absorb"]),
    default="spread",
    help="Employer response",
)
@click.option(
    "--employee",
    "-E",
    type=click.Choice(["maintain", "cash"]),
    default="maintain",
    help="Employee response",
)
@click.option(
    "--year",
    "-y",
    default=2026,
    type=int,
    help="Year to analyze",
)
@click.option(
    "--dataset",
    "-d",
    default=None,
    type=click.Path(exists=True),
    help="Path to dataset file",
)
def run_single(cap, employer, employee, year, dataset):
    """Run a single scenario."""
    from salary_sacrifice.analysis import run_scenario
    from salary_sacrifice.reforms import (
        EmployeeResponse,
        EmployerResponse,
        SalarySacrificeCapScenario,
    )

    employer_map = {
        "spread": EmployerResponse.SPREAD_COST,
        "absorb": EmployerResponse.ABSORB_COST,
    }
    employee_map = {
        "maintain": EmployeeResponse.MAINTAIN_PENSION,
        "cash": EmployeeResponse.TAKE_CASH,
    }

    scenario = SalarySacrificeCapScenario(
        cap_amount=cap,
        employer_response=employer_map[employer],
        employee_response=employee_map[employee],
    )

    click.echo(f"Running scenario: {scenario.name}")
    click.echo()

    results = run_scenario(scenario, years=[year], dataset_path=dataset)

    click.echo("=" * 50)
    click.echo(f"Revenue impact ({year}): £{results['revenue_by_year'][year]:.2f}bn")
    click.echo(
        f"Affected workers: {results['affected_population']['affected_workers']:,.0f}"
    )
    click.echo("=" * 50)


@click.command()
@click.option(
    "--cap",
    "-c",
    default=2000,
    type=float,
    help="Cap amount in GBP",
)
@click.option(
    "--year",
    "-y",
    default=2026,
    type=int,
    help="Year to analyze",
)
@click.option(
    "--dataset",
    "-d",
    default=None,
    type=click.Path(exists=True),
    help="Path to dataset file",
)
def distributional(cap, year, dataset):
    """Calculate distributional impact by income decile."""
    from salary_sacrifice.analysis import calculate_distributional_impact
    from salary_sacrifice.reforms import SalarySacrificeCapScenario

    scenario = SalarySacrificeCapScenario(cap_amount=cap)

    click.echo(f"Calculating distributional impact for £{cap:,.0f} cap...")
    results = calculate_distributional_impact(scenario, year=year, dataset_path=dataset)

    click.echo()
    click.echo("=" * 70)
    click.echo("DISTRIBUTIONAL IMPACT BY INCOME DECILE")
    click.echo("=" * 70)
    click.echo(f"{'Decile':<10} {'Avg Change':>15} {'% Change':>12}")
    click.echo("-" * 70)

    for _, row in results.iterrows():
        decile = int(row["decile"])
        avg_chg = row["avg_change"]
        pct_chg = row["pct_change"]
        click.echo(f"{decile:<10} £{avg_chg:>13,.2f} {pct_chg:>11,.3f}%")


main.add_command(run_matrix, "matrix")
main.add_command(run_single, "single")
main.add_command(distributional, "distributional")


if __name__ == "__main__":
    main()
