# UK Salary Sacrifice Cap Analysis

Analysis of a hypothetical UK salary sacrifice pension contribution cap using PolicyEngine.

## Overview

This package models the fiscal and distributional impacts of capping salary sacrifice pension contributions, with configurable behavioral assumptions about how employers and employees respond.

## Scenario Matrix

The model explores a 2x2 matrix of behavioral responses:

### Employer Response (to increased NI costs)
- **SPREAD_COST**: Spread the additional employer NI cost across all employees (cost-neutral for employer)
- **ABSORB_COST**: Employer absorbs the full extra NI cost

### Employee Response (to excess above cap)
- **MAINTAIN_PENSION**: Redirect excess to employee pension contributions (maintains total pension saving)
- **TAKE_CASH**: Take the excess as taxable cash instead (reduces pension saving)

This gives 4 scenarios:
1. **Spread + Maintain**: Broad haircut, full pension maintained
2. **Spread + Take Cash**: Broad haircut, lower pension saving
3. **Absorb + Maintain**: No haircut, full pension maintained (best for employee pension)
4. **Absorb + Take Cash**: No haircut, max take-home (best for employee cash)

## Installation

```bash
pip install -e ".[dev]"
```

## Usage

### Run the 2x2 scenario matrix

```bash
salary-sacrifice matrix --cap 2000 --year 2026
```

### Run a single scenario

```bash
salary-sacrifice single --cap 2000 --employer spread --employee maintain
```

### Calculate distributional impact

```bash
salary-sacrifice distributional --cap 2000 --year 2026
```

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest -v

# Format code
black .
```

## License

MIT
