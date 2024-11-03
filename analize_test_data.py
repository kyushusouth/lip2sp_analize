from pathlib import Path

from utils import Manager


def main():
    manager = Manager()
    manager.save_aggregated_test_data_df(
        Path("./wandb_data"),
        ["run_id", "method_id", "loss_weight", "kind"],
        Path("./results/test_data_agg_run_id.csv"),
    )
    manager.save_aggregated_test_data_df(
        Path("./wandb_data"),
        ["run_id", "method_id", "loss_weight", "speaker", "kind"],
        Path("./results/test_data_agg_run_id_speaker.csv"),
    )

    run_id_baseline = "q7pgvrou"
    run_id_compare = "kv601pjm"
    manager.save_csv_to_compare_methods(
        Path("./wandb_data"),
        run_id_baseline,
        run_id_compare,
        Path(f"./results/test_data_comparison/{run_id_baseline}_{run_id_compare}.csv"),
    )


if __name__ == "__main__":
    main()
