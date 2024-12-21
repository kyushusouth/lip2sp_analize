import json
import logging
import os
import time
from pathlib import Path

import matplotlib.pyplot as plt
import polars as pl
import seaborn as sns
import wandb

sns.set_style("whitegrid")
plt.rcParams["font.size"] = 16


class Manager:
    def __init__(self, logger: logging.Logger | None = None):
        self.logger = logger
        self.df = pl.read_csv("./results/data_info.csv")

    def download_from_wandb(
        self,
        project_root: str,
        table_lst: list[str],
        save_dir: Path,
        retry_upper_limit: int,
        retry_wait_time: float,
    ) -> None:
        assert self.logger is not None
        for row in self.df.iter_rows(named=True):
            for table in table_lst:
                if row["is_failed"] and table == "test_data":
                    continue
                if row["run_id"] not in [
                    "ptu4dbb7",
                    "lbbt90st",
                    "wa80rvxi",
                ]:
                    continue
                self.logger.info(f"Start: {row['run_id']}, {table}")
                cnt = 0
                while True:
                    try:
                        self.logger.info(f"Count: {cnt}")
                        api = wandb.Api()
                        run = api.run(f"{project_root}/{row['run_id']}")
                        df = run.history()

                        run.file(df[table].dropna().iloc[-1]["path"]).download(
                            root=str(save_dir / row["run_id"])
                        )

                        saved_path = (
                            save_dir
                            / row["run_id"]
                            / run.file(df[table].dropna().iloc[-1]["path"]).name
                        )
                        assert saved_path.exists(), "saved_path was not found."

                        if table == "test_data":
                            with open(str(saved_path), "r", encoding="utf-8") as f:
                                data = json.load(f)
                            df = pl.DataFrame(
                                data["data"], schema=data["columns"], orient="row"
                            )
                            assert df.shape == (636, 27), f"{df.shape=}"
                        else:
                            with open(str(saved_path), "r") as f:
                                data = json.load(f)["data"]
                            df = pl.DataFrame(
                                data=data,
                                schema=["epoch", "type", "value"],
                                orient="row",
                            )
                            assert (
                                df.group_by(["type"]).len().select(["len"]).n_unique()
                                == 1
                            ), f"{df.group_by(["type"]).len().select(["len"]).n_unique()=}"
                            assert (
                                int(
                                    df.select(pl.col("epoch"))
                                    .max()
                                    .to_numpy()
                                    .reshape(-1)[0]
                                )
                                == row["expected_epoch"]
                            ), f"{int(df.select(pl.col("epoch")).max().to_numpy().reshape(-1)[0])=}"

                        self.logger.info("Succeeded")
                        break

                    except IndexError as e:
                        self.logger.error(f"IndexError: {e}")
                        cnt += 1
                        if cnt > retry_upper_limit:
                            self.logger.critical("Failed")
                            break
                        time.sleep(retry_wait_time)

                    except AssertionError as e:
                        self.logger.error(f"AssertionError: {e}")
                        cnt += 1
                        os.remove(str(saved_path))
                        if cnt > retry_upper_limit:
                            self.logger.critical("Failed")
                            break
                        time.sleep(retry_wait_time)

    def preprocess_losses_df(self, data_dir: Path) -> pl.DataFrame:
        data_lst = []
        for df_row in self.df.iter_rows(named=True):
            data_path_lst = list((data_dir / df_row["run_id"]).glob("**/*.json"))
            for data_path in data_path_lst:
                loss_name = data_path.stem.split("_table_")[0]

                if df_row["method_id"] == 0 and loss_name not in [
                    "mel_loss",
                    "ssl_feature_cluster_loss",
                    "total_loss",
                ]:
                    continue
                if df_row["method_id"] == 1 and loss_name not in [
                    "mel_loss",
                    "ssl_feature_cluster_loss",
                    "ssl_conv_feature_loss",
                    "total_loss",
                ]:
                    continue
                if df_row["method_id"] == 8 and loss_name not in [
                    "ssl_conv_feature_loss",
                    "total_loss",
                ]:
                    continue
                if df_row["method_id"] in [
                    2,
                    4,
                    6,
                    7,
                    9,
                    10,
                    11,
                    12,
                ] and loss_name not in [
                    "mel_speech_ssl_loss",
                    "ssl_feature_cluster_speech_ssl_loss",
                    "total_loss",
                ]:
                    continue
                if df_row["method_id"] in [3, 5] and loss_name not in [
                    "mel_ensemble_loss",
                    "ssl_feature_cluster_ensemble_loss",
                    "total_loss",
                ]:
                    continue

                if df_row["method_id"] in [2, 4, 6, 7, 9, 10, 11, 12]:
                    if loss_name == "mel_speech_ssl_loss":
                        loss_name = "mel_loss"
                    elif loss_name == "ssl_feature_cluster_speech_ssl_loss":
                        loss_name = "ssl_feature_cluster_loss"
                elif df_row["method_id"] in [3, 5]:
                    if loss_name == "mel_ensemble_loss":
                        loss_name = "mel_loss"
                    elif loss_name == "ssl_feature_cluster_ensemble_loss":
                        loss_name = "ssl_feature_cluster_loss"

                with open(str(data_path), "r") as f:
                    data = json.load(f)["data"]

                data_required = []
                for data_row in data:
                    if df_row["run_id"] in ["2zoxvtrk", "tg4r230f"]:
                        if df_row["run_id"] == "2zoxvtrk":
                            if data_row[0] > 22:
                                continue
                            data_row += [
                                "tg4r230f",
                                loss_name,
                                df_row["method_id"],
                                df_row["loss_weight"],
                                self.df.filter((pl.col("run_id") == "tg4r230f"))
                                .select(["selected_epoch"])
                                .to_numpy()
                                .reshape(-1)[0],
                            ]
                        elif df_row["run_id"] == "tg4r230f":
                            data_row[0] += 23
                            data_row += [
                                df_row["run_id"],
                                loss_name,
                                df_row["method_id"],
                                df_row["loss_weight"],
                                df_row["selected_epoch"],
                            ]
                    elif df_row["run_id"] in ["pstdmqnu", "yz0nwsvz"]:
                        if df_row["run_id"] == "pstdmqnu":
                            if data_row[0] > 8:
                                continue
                            data_row += [
                                "yz0nwsvz",
                                loss_name,
                                df_row["method_id"],
                                df_row["loss_weight"],
                                self.df.filter((pl.col("run_id") == "yz0nwsvz"))
                                .select(["selected_epoch"])
                                .to_numpy()
                                .reshape(-1)[0],
                            ]
                        elif df_row["run_id"] == "yz0nwsvz":
                            data_row[0] += 9
                            data_row += [
                                df_row["run_id"],
                                loss_name,
                                df_row["method_id"],
                                df_row["loss_weight"],
                                df_row["selected_epoch"],
                            ]
                    elif df_row["run_id"] in ["sitbur40", "r26chnzo"]:
                        if df_row["run_id"] == "sitbur40":
                            if data_row[0] > 24:
                                continue
                            data_row += [
                                "r26chnzo",
                                loss_name,
                                df_row["method_id"],
                                df_row["loss_weight"],
                                self.df.filter((pl.col("run_id") == "r26chnzo"))
                                .select(["selected_epoch"])
                                .to_numpy()
                                .reshape(-1)[0],
                            ]
                        elif df_row["run_id"] == "r26chnzo":
                            data_row[0] += 25
                            data_row += [
                                df_row["run_id"],
                                loss_name,
                                df_row["method_id"],
                                df_row["loss_weight"],
                                df_row["selected_epoch"],
                            ]
                    else:
                        data_row += [
                            df_row["run_id"],
                            loss_name,
                            df_row["method_id"],
                            df_row["loss_weight"],
                            df_row["selected_epoch"],
                        ]
                    data_required.append(data_row)
                data_lst += data_required

        df = pl.DataFrame(
            data=data_lst,
            schema=[
                "epoch",
                "loss_type",
                "loss_value",
                "run_id",
                "loss_name",
                "method_id",
                "loss_weight",
                "selected_epoch",
            ],
            orient="row",
        )
        return df

    def save_validation_losses_csv(self, data_dir: Path, save_path: Path) -> None:
        df = self.preprocess_losses_df(data_dir)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        df.filter(
            (pl.col("epoch") == pl.col("selected_epoch"))
            & (pl.col("loss_type") == "validation loss")
        ).select(
            "run_id",
            "method_id",
            "loss_weight",
            "loss_type",
            "loss_name",
            "selected_epoch",
            "loss_value",
        ).sort(["loss_weight", "method_id", "loss_type", "loss_name"]).pivot(
            on=["loss_name"],
            index=["run_id", "method_id", "loss_weight", "selected_epoch"],
            values=["loss_value"],
        ).write_csv(str(save_path))

    def save_learning_curves(self, data_dir: Path, save_dir: Path) -> None:
        df = self.preprocess_losses_df(data_dir)
        color_lst = ["black", "red", "gold", "green", "blue", "purple"]
        loss_name_lst = [
            "mel_loss",
            "ssl_feature_cluster_loss",
            "ssl_conv_feature_loss",
            "total_loss",
        ]
        for method_id in sorted(
            list(df.select(pl.col("method_id")).unique().to_numpy().reshape(-1))
        ):
            save_dir_method = save_dir / str(method_id)
            save_dir_method.mkdir(parents=True, exist_ok=True)

            for loss_name in loss_name_lst:
                if loss_name == "ssl_conv_feature_loss" and method_id not in [1, 8]:
                    continue
                if loss_name != "ssl_conv_feature_loss" and method_id == 8:
                    continue

                fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 4))

                for loss_weight, color in zip(
                    sorted(
                        list(
                            df.select(pl.col("loss_weight"))
                            .unique()
                            .to_numpy()
                            .reshape(-1)
                        )
                    ),
                    color_lst,
                ):
                    if loss_weight == 0 and method_id != 8:
                        continue
                    if method_id == 8 and loss_weight != 0:
                        continue
                    if method_id in [10, 11, 12] and loss_weight != 0.1:
                        continue

                    data_val = df.filter(
                        (pl.col("loss_name") == loss_name)
                        & (pl.col("loss_type") == "validation loss")
                        & (pl.col("method_id") == method_id)
                        & (pl.col("loss_weight") == loss_weight)
                    ).sort(["epoch"])
                    ax.plot(
                        data_val["epoch"],
                        data_val["loss_value"],
                        label=str(loss_weight) + "_val",
                        color=color,
                    )

                    data_train = df.filter(
                        (pl.col("loss_name") == loss_name)
                        & (pl.col("loss_type") == "train loss")
                        & (pl.col("method_id") == method_id)
                        & (pl.col("loss_weight") == loss_weight)
                    ).sort(["epoch"])
                    ax.plot(
                        data_train["epoch"],
                        data_train["loss_value"],
                        label=str(loss_weight) + "_train",
                        color=color,
                        linestyle="dotted",
                    )

                    val_loss_selected_epoch = df.filter(
                        (pl.col("loss_name") == loss_name)
                        & (pl.col("loss_type") == "validation loss")
                        & (pl.col("method_id") == method_id)
                        & (pl.col("loss_weight") == loss_weight)
                        & (pl.col("epoch") == pl.col("selected_epoch"))
                    )
                    assert (
                        val_loss_selected_epoch.shape[0] == 1
                    ), f"{method_id=}, {loss_name=}, {loss_weight=}, {val_loss_selected_epoch.shape=}"
                    ax.plot(
                        val_loss_selected_epoch["epoch"],
                        val_loss_selected_epoch["loss_value"],
                        marker="o",
                        color=color,
                        markersize=8,
                    )

                    ax.set_xlabel("Epoch")
                    ax.set_ylabel("Loss")
                    ax.set_xlim(-1, 51)
                    if loss_name == "mel_loss":
                        if method_id == 7:
                            ax.set_ylim(0.35, 3.5)
                        else:
                            ax.set_ylim(0.35, 1.0)
                    elif loss_name == "ssl_feature_cluster_loss":
                        if method_id == 7:
                            ax.set_ylim(0.5, 33.0)
                        else:
                            ax.set_ylim(0.5, 5.5)
                    elif loss_name == "ssl_conv_feature_loss":
                        ax.set_ylim(0.1, 0.4)
                    elif loss_name == "total_loss":
                        ax.set_ylim(0, 4.0)

                plt.legend(bbox_to_anchor=(1, 1), loc="upper left")
                fig.tight_layout()

                save_path = save_dir / str(method_id) / f"{loss_name}.png"
                save_path.parent.mkdir(exist_ok=True, parents=True)
                fig.savefig(str(save_path))
                plt.close()

        loss_weight = 0.1
        for loss_name in loss_name_lst:
            if loss_name == "ssl_conv_feature_loss":
                continue

            fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 4))

            for method_id, color, learning_rate in zip(
                [10, 11, 12], ["red", "gold", "green"], ["5.0e-4", "5.0e-5", "5.0e-6"]
            ):
                data_val = df.filter(
                    (pl.col("loss_name") == loss_name)
                    & (pl.col("loss_type") == "validation loss")
                    & (pl.col("method_id") == method_id)
                    & (pl.col("loss_weight") == loss_weight)
                ).sort(["epoch"])
                ax.plot(
                    data_val["epoch"],
                    data_val["loss_value"],
                    label=str(learning_rate) + "_val",
                    color=color,
                )

                data_train = df.filter(
                    (pl.col("loss_name") == loss_name)
                    & (pl.col("loss_type") == "train loss")
                    & (pl.col("method_id") == method_id)
                    & (pl.col("loss_weight") == loss_weight)
                ).sort(["epoch"])
                ax.plot(
                    data_train["epoch"],
                    data_train["loss_value"],
                    label=str(learning_rate) + "_train",
                    color=color,
                    linestyle="dotted",
                )

                val_loss_selected_epoch = df.filter(
                    (pl.col("loss_name") == loss_name)
                    & (pl.col("loss_type") == "validation loss")
                    & (pl.col("method_id") == method_id)
                    & (pl.col("loss_weight") == loss_weight)
                    & (pl.col("epoch") == pl.col("selected_epoch"))
                )
                assert (
                    val_loss_selected_epoch.shape[0] == 1
                ), f"{method_id=}, {loss_name=}, {loss_weight=}, {val_loss_selected_epoch.shape=}"
                ax.plot(
                    val_loss_selected_epoch["epoch"],
                    val_loss_selected_epoch["loss_value"],
                    marker="o",
                    color=color,
                    markersize=8,
                )

                ax.set_xlabel("Epoch")
                ax.set_ylabel("Loss")
                ax.set_xlim(-1, 51)
                if loss_name == "mel_loss":
                    ax.set_ylim(0.2, 1.0)
                elif loss_name == "ssl_feature_cluster_loss":
                    ax.set_ylim(0.5, 5.5)
                elif loss_name == "ssl_conv_feature_loss":
                    ax.set_ylim(0.1, 0.4)
                elif loss_name == "total_loss":
                    ax.set_ylim(0, 4.0)

            plt.legend(bbox_to_anchor=(1, 1), loc="upper left")
            fig.tight_layout()

            save_path = save_dir / "10_11_12" / f"{loss_name}.png"
            save_path.parent.mkdir(exist_ok=True, parents=True)
            fig.savefig(str(save_path))
            plt.close()

    def save_learning_curves_compare(self, data_dir: Path, save_dir: Path) -> None:
        df = self.preprocess_losses_df(data_dir)
        loss_name_lst = [
            "mel_loss",
            "ssl_feature_cluster_loss",
            "total_loss",
        ]
        for loss_name in loss_name_lst:
            fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 4))

            data_val = df.filter(
                (pl.col("loss_name") == loss_name)
                & (pl.col("loss_type") == "validation loss")
                & (pl.col("method_id") == 2)
                & (pl.col("loss_weight") == 0.1)
            ).sort(["epoch"])
            ax.plot(
                data_val["epoch"],
                data_val["loss_value"],
                label="B_Randomized" + "_val",
                color="blue",
            )

            data_train = df.filter(
                (pl.col("loss_name") == loss_name)
                & (pl.col("loss_type") == "train loss")
                & (pl.col("method_id") == 2)
                & (pl.col("loss_weight") == 0.1)
            ).sort(["epoch"])
            ax.plot(
                data_train["epoch"],
                data_train["loss_value"],
                label="B_Randomized" + "_train",
                color="blue",
                linestyle="dotted",
            )

            val_loss_selected_epoch = df.filter(
                (pl.col("loss_name") == loss_name)
                & (pl.col("loss_type") == "validation loss")
                & (pl.col("method_id") == 2)
                & (pl.col("loss_weight") == 0.1)
                & (pl.col("epoch") == pl.col("selected_epoch"))
            )
            assert val_loss_selected_epoch.shape[0] == 1
            ax.plot(
                val_loss_selected_epoch["epoch"],
                val_loss_selected_epoch["loss_value"],
                marker="o",
                color="blue",
                markersize=8,
            )

            data_val = df.filter(
                (pl.col("loss_name") == loss_name)
                & (pl.col("loss_type") == "validation loss")
                & (pl.col("method_id") == 4)
                & (pl.col("loss_weight") == 0.1)
            ).sort(["epoch"])
            ax.plot(
                data_val["epoch"],
                data_val["loss_value"],
                label="B_Pretrained" + "_val",
                color="red",
            )

            data_train = df.filter(
                (pl.col("loss_name") == loss_name)
                & (pl.col("loss_type") == "train loss")
                & (pl.col("method_id") == 4)
                & (pl.col("loss_weight") == 0.1)
            ).sort(["epoch"])
            ax.plot(
                data_train["epoch"],
                data_train["loss_value"],
                label="B_Pretrained" + "_train",
                color="red",
                linestyle="dotted",
            )

            val_loss_selected_epoch = df.filter(
                (pl.col("loss_name") == loss_name)
                & (pl.col("loss_type") == "validation loss")
                & (pl.col("method_id") == 4)
                & (pl.col("loss_weight") == 0.1)
                & (pl.col("epoch") == pl.col("selected_epoch"))
            )
            assert val_loss_selected_epoch.shape[0] == 1
            ax.plot(
                val_loss_selected_epoch["epoch"],
                val_loss_selected_epoch["loss_value"],
                marker="o",
                color="red",
                markersize=8,
            )

            ax.set_xlabel("Epoch")
            ax.set_ylabel("Loss")
            ax.set_xlim(-1, 51)
            if loss_name == "mel_loss":
                ax.set_ylim(0.4, 0.8)
            elif loss_name == "ssl_feature_cluster_loss":
                ax.set_ylim(0.5, 5.5)
            elif loss_name == "ssl_conv_feature_loss":
                ax.set_ylim(0.1, 0.4)
            elif loss_name == "total_loss":
                ax.set_ylim(0, 4.0)

            plt.legend(bbox_to_anchor=(1, 1), loc="upper left")
            fig.tight_layout()

            save_path = save_dir / "compare_2_4" / f"{loss_name}.png"
            save_path.parent.mkdir(exist_ok=True, parents=True)
            fig.savefig(str(save_path))
            plt.close()

    def preprocess_test_data_df(self, data_dir: Path) -> pl.DataFrame:
        data_path_lst = list(data_dir.glob("**/test_data_*.json"))
        df_lst = []
        for data_path in data_path_lst:
            run_id = data_path.parents[2].name
            if (
                self.df.filter((pl.col("run_id") == run_id))
                .select(["is_failed"])
                .to_numpy()
                .reshape(-1)[0]
            ):
                continue
            with open(str(data_path), "r", encoding="utf-8") as f:
                data = json.load(f)
            df = pl.DataFrame(data["data"], schema=data["columns"], orient="row")
            df = df.with_columns(pl.lit(run_id).alias("run_id"))
            df_lst.append(df)

        df_result = pl.concat(df_lst, how="vertical")
        df_result = df_result.with_columns(
            pl.col("wer_kanjikana") * 100,
            pl.col("cer_kana") * 100,
            pl.col("per_phoneme") * 100,
        )
        return df_result

    def aggregate_test_data_df(
        self, df: pl.DataFrame, cols_groupby: list[str]
    ) -> pl.DataFrame:
        df_result = (
            df.join(
                self.df.select(["run_id", "method_id", "loss_weight"]),
                on=["run_id"],
                how="left",
                coalesce=True,
            )
            .group_by(cols_groupby)
            .agg(
                pl.col("spk_sim").mean(),
                (
                    (
                        pl.col("delt_kanjikana").sum()
                        + pl.col("inst_kanjikana").sum()
                        + pl.col("subst_kanjikana").sum()
                    )
                    / pl.col("utt_gt_kanjikana").str.split(" ").list.len().sum()
                    * 100
                ).alias("wer_kanjikana"),
                (
                    (
                        pl.col("delt_kana").sum()
                        + pl.col("inst_kana").sum()
                        + pl.col("subst_kana").sum()
                    )
                    / pl.col("utt_gt_kana").str.split(" ").list.len().sum()
                    * 100
                ).alias("cer_kana"),
                (
                    (
                        pl.col("delt_phoneme").sum()
                        + pl.col("inst_phoneme").sum()
                        + pl.col("subst_phoneme").sum()
                    )
                    / pl.col("utt_gt_phoneme").str.split(" ").list.len().sum()
                    * 100
                ).alias("per_phoneme"),
            )
            .select(
                cols_groupby
                + [
                    "wer_kanjikana",
                    "cer_kana",
                    "per_phoneme",
                    "spk_sim",
                ]
            )
        )
        return df_result

    def save_aggregated_test_data_df(
        self, data_dir: Path, cols_groupby: list[str], save_path: Path
    ) -> None:
        df = self.preprocess_test_data_df(data_dir)
        df_agg = self.aggregate_test_data_df(df, cols_groupby)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        df_agg.write_csv(str(save_path))

    def save_csv_to_compare_methods(
        self,
        data_dir: Path,
        run_id_baseline: str,
        run_id_compare: str,
        save_path: Path,
    ) -> None:
        df = self.preprocess_test_data_df(data_dir)

        df_result = (
            df.filter(
                (pl.col("run_id") == run_id_compare)
                & (pl.col("kind") == "pred_mel_speech_ssl")
            )
            .join(self.df.select(["run_id", "run_date"]), on=["run_id"], how="left")
            .select(
                [
                    "run_date",
                    "speaker",
                    "filename",
                    "wer_kanjikana",
                    "utt_gt_kanjikana",
                    "utt_recog_kanjikana",
                    "cer_kana",
                    "utt_gt_kana",
                    "utt_recog_kana",
                    "per_phoneme",
                    "utt_gt_phoneme",
                    "utt_recog_phoneme",
                    "spk_sim",
                ]
            )
            .rename(
                {
                    "run_date": "run_date_compare",
                    "wer_kanjikana": "wer_kanjikana_compare",
                    "utt_recog_kanjikana": "utt_recog_kanjikana_compare",
                    "cer_kana": "cer_kana_compare",
                    "utt_recog_kana": "utt_recog_kana_compare",
                    "per_phoneme": "per_phoneme_compare",
                    "utt_recog_phoneme": "utt_recog_phoneme_compare",
                    "spk_sim": "spk_sim_compare",
                }
            )
            .join(
                df.filter(
                    (pl.col("run_id") == run_id_baseline)
                    & (pl.col("kind") == "pred_mel_speech_ssl")
                )
                .join(self.df.select(["run_id", "run_date"]), on=["run_id"], how="left")
                .select(
                    [
                        "run_date",
                        "speaker",
                        "filename",
                        "wer_kanjikana",
                        "utt_recog_kanjikana",
                        "cer_kana",
                        "utt_recog_kana",
                        "per_phoneme",
                        "utt_recog_phoneme",
                        "spk_sim",
                    ]
                )
                .rename(
                    {
                        "run_date": "run_date_baseline",
                        "wer_kanjikana": "wer_kanjikana_baseline",
                        "utt_recog_kanjikana": "utt_recog_kanjikana_baseline",
                        "cer_kana": "cer_kana_baseline",
                        "utt_recog_kana": "utt_recog_kana_baseline",
                        "per_phoneme": "per_phoneme_baseline",
                        "utt_recog_phoneme": "utt_recog_phoneme_baseline",
                        "spk_sim": "spk_sim_baseline",
                    }
                ),
                on=["speaker", "filename"],
                how="left",
            )
            .with_columns(
                (
                    pl.col("wer_kanjikana_compare") - pl.col("wer_kanjikana_baseline")
                ).alias("wer_kanjikana_diff"),
                (pl.col("cer_kana_compare") - pl.col("cer_kana_baseline")).alias(
                    "cer_kana_diff"
                ),
                (pl.col("per_phoneme_compare") - pl.col("per_phoneme_baseline")).alias(
                    "per_phoneme_diff"
                ),
                (pl.col("spk_sim_compare") - pl.col("spk_sim_baseline")).alias(
                    "spk_sim_diff"
                ),
            )
            .select(
                [
                    "run_date_baseline",
                    "run_date_compare",
                    "speaker",
                    "filename",
                    "wer_kanjikana_baseline",
                    "wer_kanjikana_compare",
                    "wer_kanjikana_diff",
                    "utt_gt_kanjikana",
                    "utt_recog_kanjikana_baseline",
                    "utt_recog_kanjikana_compare",
                    "cer_kana_baseline",
                    "cer_kana_compare",
                    "cer_kana_diff",
                    "utt_gt_kana",
                    "utt_recog_kana_baseline",
                    "utt_recog_kana_compare",
                    "per_phoneme_baseline",
                    "per_phoneme_compare",
                    "per_phoneme_diff",
                    "utt_gt_phoneme",
                    "utt_recog_phoneme_baseline",
                    "utt_recog_phoneme_compare",
                    "spk_sim_baseline",
                    "spk_sim_compare",
                    "spk_sim_diff",
                ],
            )
            .sort(["speaker", "filename"])
        )

        save_path.parent.mkdir(parents=True, exist_ok=True)
        df_result.write_csv(str(save_path))
