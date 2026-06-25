from pathlib import Path

from simulation.pipeline import run_orbit_pipeline


def main() -> None:
    output_dir = Path("outputs")
    run_orbit_pipeline(output_dir)


if __name__ == "__main__":
    main()
