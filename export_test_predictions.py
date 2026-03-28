import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import DataProcessMoudle
import Net
import decode
import videoAugmentation
from ReadConfig import readConfig
from WER import WerList


def indices_to_text(indices, idx2word):
    tokens = []
    for x in indices:
        try:
            v = int(x)
        except Exception:
            continue
        if 0 <= v < len(idx2word):
            tokens.append(idx2word[v])
    return "".join(tokens), tokens


def pred_batch_to_indices(pred, word2idx):
    pred_batch_indices = []
    if not pred:
        return pred_batch_indices
    for pred_item in pred:
        sample_indices = []
        if isinstance(pred_item, list):
            for token in pred_item:
                word = token[0] if isinstance(token, tuple) else token
                if word in word2idx:
                    sample_indices.append(word2idx[word])
        pred_batch_indices.append(sample_indices)
    return pred_batch_indices


def build_test_loader(config_params, word2idx):
    transform_test = videoAugmentation.Compose([
        videoAugmentation.CenterCrop(224),
        videoAugmentation.ToTensor(),
    ])
    test_data = DataProcessMoudle.MyDataset(
        config_params["testDataPath"],
        config_params["testLabelPath"],
        word2idx,
        config_params["dataSetName"],
        transform=transform_test,
        frameSampleStride=max(1, int(config_params.get("frameSampleStride", 1))),
        preprocessedRoot=config_params.get("preprocessedRoot", "CSL/preprocessed"),
        usePreprocessed=int(config_params.get("usePreprocessed", 0)),
    )
    test_loader = DataLoader(
        dataset=test_data,
        batch_size=1,
        shuffle=False,
        num_workers=int(config_params["numWorkers"]),
        pin_memory=bool(int(config_params["pinmMemory"])),
        collate_fn=DataProcessMoudle.collate_fn,
        drop_last=False,
    )
    return test_loader


def load_model(config_params, vocab_size):
    module_net = Net.moduleNet(
        int(config_params["hiddenSize"]),
        vocab_size + 1,
        config_params["moduleChoice"],
        config_params["device"],
        config_params["dataSetName"],
        True,
        cnnChunkSize=max(1, int(config_params.get("cnnChunkSize", 64))),
    ).to(config_params["device"])

    checkpoint_path = config_params["bestModuleSavePath"]
    checkpoint = torch.load(checkpoint_path, map_location=torch.device("cpu"))
    module_net.load_state_dict(checkpoint["moduleNet_state_dict"])
    module_net.eval()
    return module_net, checkpoint_path


def main():
    config_params = readConfig()
    word2idx, word_set_num, idx2word = DataProcessMoudle.Word2Id(
        config_params["trainLabelPath"],
        config_params["validLabelPath"],
        config_params["testLabelPath"],
        config_params["dataSetName"],
        trainDataPath=config_params["trainDataPath"],
        validDataPath=config_params["validDataPath"],
        testDataPath=config_params["testDataPath"],
    )
    print(f"export vocab size(without blank): {word_set_num}")

    test_loader = build_test_loader(config_params, word2idx)
    module_net, checkpoint_path = load_model(config_params, word_set_num)
    log_softmax = nn.LogSoftmax(dim=-1)
    decoder = decode.Decode(word2idx, word_set_num + 1, 'beam')

    out_dir = "test_reports"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "test_predictions_full.md")

    total_err = 0
    total_ref = 0
    total_del = 0
    total_ins = 0
    total_sub = 0

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("# Test Prediction Report\n\n")
        f.write(f"- checkpoint: `{checkpoint_path}`\n")
        f.write(f"- test split: `{config_params['testDataPath']}`\n")
        f.write(f"- vocab size (without blank): {word_set_num}\n\n")
        f.write("| # | sample | raw_len | lgt | ref | hyp | wer | D | I | S |\n")
        f.write("|---:|---|---:|---:|---|---|---:|---:|---:|---:|\n")

        for idx, batch in enumerate(test_loader, start=1):
            data = batch["video"].to(config_params["device"])
            label = batch["label"]
            data_len = batch["videoLength"]
            info = batch["info"]

            target_data = [yi.clone().detach().to(config_params["device"]) for yi in label]

            with torch.no_grad():
                log_probs1, _, _, _, _, lgt, _, _, _ = module_net(data, data_len, False)
                log_probs1 = log_softmax(log_probs1)
                pred, _ = decoder.decode(log_probs1, lgt, batch_first=False, probs=False)

            pred_batch_indices = pred_batch_to_indices(pred, word2idx)
            pred_indices = pred_batch_indices[0] if pred_batch_indices else []
            ref_indices = target_data[0].tolist() if hasattr(target_data[0], "tolist") else target_data[0]

            ref_text, ref_tokens = indices_to_text(ref_indices, idx2word)
            hyp_text, hyp_tokens = indices_to_text(pred_indices, idx2word)
            wer = WerList([" ".join(ref_tokens)], [" ".join(hyp_tokens)])

            total_err += wer["wer"] * max(1, len(ref_tokens)) / 100.0
            total_ref += len(ref_tokens)
            total_del += wer["del_rate"] * max(1, len(ref_tokens)) / 100.0
            total_ins += wer["ins_rate"] * max(1, len(ref_tokens)) / 100.0
            total_sub += wer["sub_rate"] * max(1, len(ref_tokens)) / 100.0

            raw_len = int(data_len[0].item()) if torch.is_tensor(data_len[0]) else int(data_len[0])
            eff_len = int(lgt[0].item()) if torch.is_tensor(lgt[0]) else int(lgt[0])
            sample_name = info[0] if info else f"sample_{idx}"

            f.write(
                f"| {idx} | {sample_name} | {raw_len} | {eff_len} | {ref_text} | {hyp_text} | "
                f"{wer['wer']:.2f} | {wer['del_rate']:.2f} | {wer['ins_rate']:.2f} | {wer['sub_rate']:.2f} |\n"
            )

        total_wer = (total_err / total_ref * 100.0) if total_ref > 0 else 0.0
        total_del_rate = (total_del / total_ref * 100.0) if total_ref > 0 else 0.0
        total_ins_rate = (total_ins / total_ref * 100.0) if total_ref > 0 else 0.0
        total_sub_rate = (total_sub / total_ref * 100.0) if total_ref > 0 else 0.0

        f.write("\n## Summary\n\n")
        f.write(f"- total samples: {len(test_loader)}\n")
        f.write(f"- total ref tokens: {total_ref}\n")
        f.write(f"- WER: {total_wer:.2f}\n")
        f.write(f"- DEL: {total_del_rate:.2f}\n")
        f.write(f"- INS: {total_ins_rate:.2f}\n")
        f.write(f"- SUB: {total_sub_rate:.2f}\n")

    print(f"Test prediction report saved to: {out_path}")


if __name__ == "__main__":
    main()
