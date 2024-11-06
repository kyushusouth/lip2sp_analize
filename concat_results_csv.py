import polars as pl


def main():
    df_test_data = pl.read_csv("./results/test_data_agg_run_id.csv")
    df_val_loss = pl.read_csv("./results/validations_losses.csv")
    df_val_loss = df_val_loss.with_columns(pl.lit("pred_mel_speech_ssl").alias("kind"))
    df_test_data.join(
        df_val_loss, on=["run_id", "method_id", "loss_weight", "kind"], how="left"
    ).write_csv("./results/concat_test_data_agg_run_id_validations_losses.csv")


if __name__ == "__main__":
    main()
