import logging
from datetime import datetime
from pathlib import Path

from utils import Manager


def main():
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file_path = Path(f"./logs/{current_time}.log")
    log_file_path.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    file_handler = logging.FileHandler(str(log_file_path))
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    manager = Manager(logger)
    manager.download_from_wandb(
        "minami373/lip2sp-base_hubert_2",
        [
            "mel_loss_table",
            "ssl_conv_feature_loss_table",
            "ssl_feature_cluster_loss_table",
            "mel_speech_ssl_loss_table",
            "ssl_feature_cluster_speech_ssl_loss_table",
            "mel_ensemble_loss_table",
            "ssl_feature_cluster_ensemble_loss_table",
            "total_loss_table",
            "test_data",
        ],
        Path("./wandb_data_test"),
        100,
        3.0,
    )


if __name__ == "__main__":
    main()
