from pathlib import Path

import polars as pl


def main():
    save_path = Path("./results/data_info.csv")
    save_path.parent.mkdir(parents=True, exist_ok=True)

    run_lst = [
        ["dca2gb3d", "5", "1.0", "20241031_233821", False],
        ["cdyv6fps", "4", "1.0", "20241031_230230", False],
        ["wlxrw27k", "3", "1.0", "20241031_222413", False],
        ["5xn74zem", "2", "1.0", "20241031_214046", False],
        ["zhoeivjb", "1", "1.0", "20241031_204425", False],
        ["sosf0acz", "0", "1.0", "20241031_195052", False],
        ["2f694nev", "5", "0.1", "20241028_223025", False],
        ["h71pe66b", "4", "0.1", "20241028_214954", False],
        ["4h4id4tz", "3", "0.1", "20241028_210910", False],
        ["wy0h8bpu", "2", "0.1", "20241028_200713", False],
        ["o6hptbk5", "1", "0.1", "20241028_184008", False],
        ["285wqmad", "0", "0.1", "20241028_172128", False],
        ["1o40lyod", "5", "0.01", "20241028_161118", False],
        ["tg4r230f", "4", "0.01", "20241028_154643", False],
        ["2zoxvtrk", "4", "0.01", "20241028_150238", True],
        ["uw487j74", "3", "0.01", "20241028_141442", False],
        ["2njl6ce3", "2", "0.01", "20241028_131108", False],
        ["yz0nwsvz", "1", "0.01", "20241028_113232", False],
        ["pstdmqnu", "1", "0.01", "20241028_111049", True],
        ["ukfz2470", "0", "0.01", "20241028_091345", False],
        ["lic1d8lj", "5", "0.001", "20241028_080415", False],
        ["pvun1i3f", "4", "0.001", "20241028_070744", False],
        ["ceq1xo4e", "3", "0.001", "20241028_061843", False],
        ["3jyrz1so", "2", "0.001", "20241028_052606", False],
        ["e1xl19lr", "1", "0.001", "20241028_032925", False],
        ["kv601pjm", "0", "0.001", "20241028_013251", False],
        ["drpgzb7p", "5", "0.0001", "20241028_003600", False],
        ["2wwcs7kw", "4", "0.0001", "20241027_233750", False],
        ["7091hf62", "3", "0.0001", "20241027_222745", False],
        ["ehyszq9v", "2", "0.0001", "20241027_210956", False],
        ["48nmuxoh", "1", "0.0001", "20241027_191342", False],
        ["q7pgvrou", "0", "0.0001", "20241027_171757", False],
        # 2で入力をメルスペクトログラムとHuBERT離散特徴量
        ["wdsnci1d", "6", "0.0001", "20241106_145624", False],
        ["56dqwrre", "6", "0.001", "20241106_171208", False],
        ["7xji36uy", "6", "0.01", "20241106_192911", False],
        ["pxeo2tih", "6", "0.1", "20241106_205822", False],
        ["x44nt08d", "6", "1.0", "20241106_235636", False],
        # 2でネットワークAとネットワークBを一気に学習
        ["q3g90r80", "7", "0.0001", "20241106_162207", False],
        ["cpurcixo", "7", "0.001", "20241106_183841", False],
        ["w7qmfdim", "7", "0.01", "20241106_201646", False],
        ["t231ink6", "7", "0.1", "20241106_214845", False],
        ["rwxbpec0", "7", "1.0", "20241107_004931", False],
        # 2でネットワークAをHuBERT中間特徴量のみを推定するように学習した場合
        ["l1pr46xw", "8", "0", "20241109_143916", False],  # A
        ["sitbur40", "9", "0.0001", "20241109_163501", True],  # B
        ["r26chnzo", "9", "0.0001", "20241109_172152", False],  # B
        ["pa5ndjr3", "9", "0.001", "20241109_180945", False],  # B
        ["vn7tt7n0", "9", "0.01", "20241109_191946", False],  # B
        ["1kpb7wfk", "9", "0.1", "20241109_201211", False],  # B
        ["vzr1la0o", "9", "1.0", "20241109_210312", False],  # B
    ]

    ckpt_path_lst = [
        Path("base_hubert_2/20241109_210312/epoch:12-step:650.ckpt"),
        Path("base_hubert_2/20241109_201211/epoch:17-step:900.ckpt"),
        Path("base_hubert_2/20241109_191946/epoch:18-step:950.ckpt"),
        Path("base_hubert_2/20241109_180945/epoch:30-step:1550.ckpt"),
        Path("base_hubert_2/20241109_172152/epoch:46-step:2350.ckpt"),
        Path("base_hubert_2/20241109_163501/epoch:24-step:1250.ckpt"),
        Path("base_hubert_2/20241109_143916/epoch:44-step:2250.ckpt"),
        Path("base_hubert_2/20241107_004931/epoch:9-step:500.ckpt"),
        Path("base_hubert_2/20241106_214845/epoch:42-step:2150.ckpt"),
        Path("base_hubert_2/20241106_201646/epoch:3-step:200.ckpt"),
        Path("base_hubert_2/20241106_183841/epoch:6-step:350.ckpt"),
        Path("base_hubert_2/20241106_162207/epoch:6-step:350.ckpt"),
        Path("base_hubert_2/20241106_235636/epoch:18-step:950.ckpt"),
        Path("base_hubert_2/20241106_205822/epoch:16-step:850.ckpt"),
        Path("base_hubert_2/20241106_192911/epoch:14-step:750.ckpt"),
        Path("base_hubert_2/20241106_171208/epoch:43-step:2200.ckpt"),
        Path("base_hubert_2/20241106_145624/epoch:43-step:2200.ckpt"),
        Path("base_hubert_2/20241027_171757/epoch:45-step:2300.ckpt"),
        Path("base_hubert_2/20241027_191342/epoch:47-step:2400.ckpt"),
        Path("base_hubert_2/20241027_210956/epoch:35-step:1800.ckpt"),
        Path("base_hubert_2/20241027_222745/epoch:49-step:2500.ckpt"),
        Path("base_hubert_2/20241027_233750/epoch:22-step:1150.ckpt"),
        Path("base_hubert_2/20241028_003600/epoch:28-step:1450.ckpt"),
        Path("base_hubert_2/20241028_013251/epoch:47-step:2400.ckpt"),
        Path("base_hubert_2/20241028_032925/epoch:45-step:2300.ckpt"),
        Path("base_hubert_2/20241028_052606/epoch:18-step:950.ckpt"),
        Path("base_hubert_2/20241028_061843/epoch:22-step:1150.ckpt"),
        Path("base_hubert_2/20241028_070744/epoch:21-step:1100.ckpt"),
        Path("base_hubert_2/20241028_080415/epoch:45-step:2300.ckpt"),
        Path("base_hubert_2/20241028_091345/epoch:47-step:2400.ckpt"),
        Path("base_hubert_2/20241028_111049/epoch:8-step:450.ckpt"),
        Path("base_hubert_2/20241028_113232/epoch:43-step:2200.ckpt"),
        Path("base_hubert_2/20241028_131108/epoch:25-step:1300.ckpt"),
        Path("base_hubert_2/20241028_141442/epoch:21-step:1100.ckpt"),
        Path("base_hubert_2/20241028_150238/epoch:22-step:1150.ckpt"),
        Path("base_hubert_2/20241028_154643/epoch:27-step:1400.ckpt"),
        Path("base_hubert_2/20241028_161118/epoch:45-step:2300.ckpt"),
        Path("base_hubert_2/20241028_172128/epoch:21-step:1100.ckpt"),
        Path("base_hubert_2/20241028_184008/epoch:25-step:1300.ckpt"),
        Path("base_hubert_2/20241028_200713/epoch:24-step:1250.ckpt"),
        Path("base_hubert_2/20241028_210910/epoch:15-step:800.ckpt"),
        Path("base_hubert_2/20241028_214954/epoch:10-step:550.ckpt"),
        Path("base_hubert_2/20241028_223025/epoch:27-step:1400.ckpt"),
        Path("base_hubert_2/20241028_232525/epoch:7-step:400.ckpt"),
        Path("base_hubert_2/20241029_001325/epoch:11-step:600.ckpt"),
        Path("base_hubert_2/20241029_011011/epoch:9-step:500.ckpt"),
        Path("base_hubert_2/20241029_014934/epoch:16-step:850.ckpt"),
        Path("base_hubert_2/20241031_195052/epoch:9-step:500.ckpt"),
        Path("base_hubert_2/20241031_204425/epoch:11-step:600.ckpt"),
        Path("base_hubert_2/20241031_214046/epoch:12-step:650.ckpt"),
        Path("base_hubert_2/20241031_222413/epoch:13-step:700.ckpt"),
        Path("base_hubert_2/20241031_230230/epoch:7-step:400.ckpt"),
        Path("base_hubert_2/20241031_233821/epoch:15-step:800.ckpt"),
    ]
    for run_data in run_lst:
        run_date = run_data[3]
        for ckpt_path in ckpt_path_lst:
            if run_date == ckpt_path.parents[0].name:
                selected_epoch = int(ckpt_path.stem.split("-")[0].replace("epoch:", ""))
                if run_date == "20241028_111049":
                    expected_epoch = 8
                elif run_date == "20241028_113232":
                    expected_epoch = 40
                elif run_date == "20241028_150238":
                    expected_epoch = 27
                elif run_date == "20241028_154643":
                    expected_epoch = 9
                else:
                    expected_epoch = min(selected_epoch + 10, 49)
                run_data.append(str(ckpt_path))
                run_data.append(selected_epoch)
                run_data.append(expected_epoch)

    df = pl.DataFrame(
        run_lst,
        schema=[
            "run_id",
            "method_id",
            "loss_weight",
            "run_date",
            "is_failed",
            "ckpt_path",
            "selected_epoch",
            "expected_epoch",
        ],
        orient="row",
    )
    df = df.sort(["method_id", "loss_weight", "run_date"])
    df.write_csv(str(save_path))


if __name__ == "__main__":
    main()
