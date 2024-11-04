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


if __name__ == "__main__":
    main()
