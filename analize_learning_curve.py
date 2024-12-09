from pathlib import Path

from utils import Manager


def main():
    manager = Manager()
    manager.save_validation_losses_csv(
        Path("./wandb_data"), Path("./results/validations_losses.csv")
    )
    manager.save_learning_curves(
        Path("./wandb_data"), Path("./results/learning_curves")
    )
    # manager.save_learning_curves_compare(
    #     Path("./wandb_data"), Path("./results/learning_curves")
    # )


if __name__ == "__main__":
    main()
