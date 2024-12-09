from pathlib import Path

from utils import Manager


def main():
    manager = Manager()
    run_id_baseline = "h71pe66b"
    run_id_compare = "wy0h8bpu"
    manager.save_csv_to_compare_methods(
        Path("./wandb_data"),
        run_id_baseline,
        run_id_compare,
        Path(f"./results/test_data_comparison/{run_id_baseline}_{run_id_compare}.csv"),
    )


if __name__ == "__main__":
    main()
