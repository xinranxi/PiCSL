import Net
import torch.nn as nn
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from WER import WerScore
import os
import DataProcessMoudle
import videoAugmentation
import numpy as np
import decode
from torch.cuda.amp import autocast as autocast
from torch.cuda.amp import GradScaler as GradScaler
from evaluation import evaluteMode
from evaluationT import evaluteModeT
import random

def seed_torch(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def stable(dataloader, seed):
    seed_torch(seed)
    return dataloader

def _pred_to_indices(pred, word2idx):
    """Convert decoder output to a flat token-id list for WerScore."""
    pred_indices = []
    if not pred:
        return pred_indices

    # pred is usually: [[(word, t), (word, t), ...]] for batch_size=1
    pred_item = pred[0]
    if isinstance(pred_item, list):
        for token in pred_item:
            word = token[0] if isinstance(token, tuple) else token
            if word in word2idx:
                pred_indices.append(word2idx[word])
    return pred_indices

def _pred_batch_to_indices(pred, word2idx):
    """Convert decoder outputs to token-id lists for each sample in batch."""
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

def _format_stats(values):
    if not values:
        return "n=0"
    arr = np.asarray(values, dtype=np.float32)
    return (
        f"n={arr.size}, min={arr.min():.1f}, p50={np.percentile(arr, 50):.1f}, "
        f"p90={np.percentile(arr, 90):.1f}, max={arr.max():.1f}, mean={arr.mean():.2f}"
    )

def _indices_to_gloss(indices, idx2word):
    tokens = []
    for x in indices:
        try:
            v = int(x)
        except:
            continue
        if 0 <= v < len(idx2word):
            tokens.append(idx2word[v])
    return " ".join(tokens)

def _blank_prob_stats(log_probs):
    blank_probs = log_probs.detach().exp()[..., 0]
    return {
        "mean": blank_probs.mean().item(),
        "min": blank_probs.min().item(),
        "max": blank_probs.max().item(),
    }

def train(configParams, isTrain=True, isCalc=False):
    # 参数初始化
    # 读入数据路径
    trainDataPath = configParams["trainDataPath"]
    validDataPath = configParams["validDataPath"]
    testDataPath = configParams["testDataPath"]
    # 读入标签路径
    trainLabelPath = configParams["trainLabelPath"]
    validLabelPath = configParams["validLabelPath"]
    testLabelPath = configParams["testLabelPath"]
    # 读入模型参数
    bestModuleSavePath = configParams["bestModuleSavePath"]
    currentModuleSavePath = configParams["currentModuleSavePath"]
    # 读入参数
    device = configParams["device"]
    hiddenSize = int(configParams["hiddenSize"])
    lr = float(configParams["lr"])
    batchSize = int(configParams["batchSize"])
    numWorkers = int(configParams["numWorkers"])
    pinmMemory = bool(int(configParams["pinmMemory"]))
    moduleChoice = configParams["moduleChoice"]
    dataSetName = configParams["dataSetName"]
    if dataSetName not in ("CE-CSL", "CSL"):
        raise ValueError(f"Unsupported dataset: {dataSetName}. Supported: CE-CSL, CSL")
    gradAccumSteps = max(1, int(configParams.get("gradAccumSteps", 8)))
    maxGradNorm = float(configParams.get("maxGradNorm", 5.0))
    freezeBackboneEpochs = max(0, int(configParams.get("freezeBackboneEpochs", 3)))
    backboneLrScale = float(configParams.get("backboneLrScale", 0.1))
    maxEpochs = max(1, int(configParams.get("maxEpochs", 55)))
    maxTrainBatches = max(0, int(configParams.get("maxTrainBatches", 0)))
    maxValidBatches = max(0, int(configParams.get("maxValidBatches", 0)))
    maxTestBatches = max(0, int(configParams.get("maxTestBatches", 0)))
    frameSampleStride = max(1, int(configParams.get("frameSampleStride", 1)))
    cnnChunkSize = max(1, int(configParams.get("cnnChunkSize", 64)))
    useAmp = bool(int(configParams.get("useAmp", 1)))
    usePreprocessed = bool(int(configParams.get("usePreprocessed", 0)))
    preprocessedRoot = configParams.get("preprocessedRoot", "CSL/preprocessed")
    videoCacheMode = configParams.get("videoCacheMode", "off")
    videoCacheFormat = configParams.get("videoCacheFormat", "npz")
    cacheTrainOnly = bool(int(configParams.get("cacheTrainOnly", 1)))
    cacheInMemoryItems = max(0, int(configParams.get("cacheInMemoryItems", 0)))
    persistentWorkers = bool(int(configParams.get("persistentWorkers", 1)))
    prefetchFactor = max(2, int(configParams.get("prefetchFactor", 2)))
    max_num_states = 1

    # 预处理语言序列
    word2idx, wordSetNum, idx2word = DataProcessMoudle.Word2Id(
        trainLabelPath, validLabelPath, testLabelPath, dataSetName,
        trainDataPath=trainDataPath, validDataPath=validDataPath, testDataPath=testDataPath
    )
    print(f"diagVocab size(without blank): {wordSetNum}, tokens: {' '.join(idx2word[1:])}")
    for save_path in [bestModuleSavePath, currentModuleSavePath, 'module/bestMoudleNet_1.pth']:
        save_dir = os.path.dirname(save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
    # 图像预处理：回退到较稳健的中等增强强度，避免强扰动导致 CTC 训练不稳定。
    transform = videoAugmentation.Compose([
        videoAugmentation.RandomCrop(224),
        videoAugmentation.RandomHorizontalFlip(0.5),
        videoAugmentation.ColorJitter(0.2, 0.2, 0.2, 0.1),
        videoAugmentation.ToTensor(),
        videoAugmentation.TemporalRescale(0.2),
    ])

    transformTest = videoAugmentation.Compose([
        videoAugmentation.CenterCrop(224),
        videoAugmentation.ToTensor(),
    ])

    # 导入数据
    trainData = DataProcessMoudle.MyDataset(
        trainDataPath, trainLabelPath, word2idx, dataSetName,
        isTrain=True, transform=transform, frameSampleStride=frameSampleStride,
        preprocessedRoot=preprocessedRoot, usePreprocessed=usePreprocessed,
        videoCacheMode=videoCacheMode, videoCacheFormat=videoCacheFormat,
        cacheTrainOnly=cacheTrainOnly, cacheInMemoryItems=cacheInMemoryItems
    )

    validData = DataProcessMoudle.MyDataset(
        validDataPath, validLabelPath, word2idx, dataSetName,
        transform=transformTest, frameSampleStride=frameSampleStride,
        preprocessedRoot=preprocessedRoot, usePreprocessed=usePreprocessed,
        videoCacheMode=videoCacheMode, videoCacheFormat=videoCacheFormat,
        cacheTrainOnly=cacheTrainOnly, cacheInMemoryItems=cacheInMemoryItems
    )

    testData = DataProcessMoudle.MyDataset(
        testDataPath, testLabelPath, word2idx, dataSetName,
        transform=transformTest, frameSampleStride=frameSampleStride,
        preprocessedRoot=preprocessedRoot, usePreprocessed=usePreprocessed,
        videoCacheMode=videoCacheMode, videoCacheFormat=videoCacheFormat,
        cacheTrainOnly=cacheTrainOnly, cacheInMemoryItems=cacheInMemoryItems
    )

    dataloader_kwargs = {
        "num_workers": numWorkers,
        "pin_memory": pinmMemory,
        "collate_fn": DataProcessMoudle.collate_fn,
    }
    if numWorkers > 0:
        dataloader_kwargs["persistent_workers"] = persistentWorkers
        dataloader_kwargs["prefetch_factor"] = prefetchFactor

    trainLoader = DataLoader(dataset=trainData, batch_size=batchSize, shuffle=True,
                             drop_last=True, **dataloader_kwargs)
    validLoader = DataLoader(dataset=validData, batch_size=1, shuffle=False,
                             drop_last=False, **dataloader_kwargs)
    testLoader = DataLoader(dataset=testData, batch_size=1, shuffle=False,
                            drop_last=False, **dataloader_kwargs)

    # 定义模型
    moduleNet = Net.moduleNet(
        hiddenSize, wordSetNum * max_num_states + 1, moduleChoice,
        device, dataSetName, True, cnnChunkSize=cnnChunkSize
    )
    moduleNet = moduleNet.to(device)

    # 损失函数定义
    PAD_IDX = 0
    if "MSTNet" == moduleChoice or "LightTFNet" == moduleChoice:
        ctcLoss = nn.CTCLoss(blank=PAD_IDX, reduction='mean', zero_infinity=True)
    elif "VAC" == moduleChoice or "CorrNet" == moduleChoice or "MAM-FSD" == moduleChoice \
         or "SEN" == moduleChoice or "TFNet" == moduleChoice:
        ctcLoss = nn.CTCLoss(blank=PAD_IDX, reduction='none', zero_infinity=True)
        kld = DataProcessMoudle.SeqKD(T=8)
        if "MAM-FSD" == moduleChoice:
            mseLoss = nn.MSELoss(reduction="mean")

    logSoftMax = nn.LogSoftmax(dim=-1)
    # 优化函数
    if moduleChoice == "LightTFNet" and hasattr(moduleNet, "conv2d"):
        backbone_params = list(moduleNet.conv2d.parameters())
        backbone_param_ids = set(map(id, backbone_params))
        other_params = [p for p in moduleNet.parameters() if id(p) not in backbone_param_ids]
        optimizer = torch.optim.Adam(
            [
                {"params": other_params, "lr": lr},
                {"params": backbone_params, "lr": lr * backboneLrScale},
            ],
            weight_decay=0.0001,
        )
    else:
        params = list(moduleNet.parameters())
        optimizer = torch.optim.Adam(params, lr=lr, weight_decay=0.0001)
    # 读取预训练模型参数
    bestLoss = 65535
    bestLossEpoch = 0
    bestWerScore = 65535
    bestWerScoreEpoch = 0
    epoch = 0

    lastEpoch = -1
    if os.path.exists(currentModuleSavePath):
        checkpoint = torch.load(currentModuleSavePath, map_location=torch.device('cpu'), weights_only=False)
        try:
            moduleNet.load_state_dict(checkpoint['moduleNet_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            bestLoss = checkpoint['bestLoss']
            bestLossEpoch = checkpoint['bestLossEpoch']
            bestWerScore = checkpoint['bestWerScore']
            bestWerScoreEpoch = checkpoint['bestWerScoreEpoch']
            epoch = checkpoint['epoch']
            lastEpoch = epoch
            print(
                f"已加载预训练模型 epoch: {epoch}, bestLoss: {bestLoss:.5f}, bestEpoch: {bestLossEpoch}, werScore: {bestWerScore:.5f}, bestEpoch: {bestWerScoreEpoch}")
        except RuntimeError as e:
            print(f"检测到 checkpoint 与当前模型结构不兼容，改为从头训练: {e}")
            bestLoss = 65535
            bestLossEpoch = 0
            bestWerScore = 65535
            bestWerScoreEpoch = 0
            epoch = 0
            lastEpoch = -1
    else:
        print(
            f"未加载预训练模型 epoch: {epoch}, bestLoss: {bestLoss}, bestEpoch: {bestLossEpoch}, werScore: {bestWerScore:.5f}, bestEpoch: {bestWerScoreEpoch}")

    # 设置学习率衰减规则
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer,
                                                     milestones=[100, 150],
                                                     gamma=0.2, last_epoch=lastEpoch)

    # 解码参数
    decoder = decode.Decode(word2idx, wordSetNum + 1, 'beam')

    if isTrain:
        print("开始训练模型")
        print(
            f"diagTrain settings: gradAccumSteps={gradAccumSteps}, "
            f"maxGradNorm={maxGradNorm}, freezeBackboneEpochs={freezeBackboneEpochs}, "
            f"backboneLrScale={backboneLrScale}"
        )
        print(
            f"diagRuntime settings: maxEpochs={maxEpochs}, maxTrainBatches={maxTrainBatches}, "
            f"maxValidBatches={maxValidBatches}, frameSampleStride={frameSampleStride}, "
            f"cnnChunkSize={cnnChunkSize}, useAmp={useAmp}, videoCacheMode={videoCacheMode}, "
            f"videoCacheFormat={videoCacheFormat}, cacheTrainOnly={cacheTrainOnly}, "
            f"cacheInMemoryItems={cacheInMemoryItems}, persistentWorkers={persistentWorkers}, "
            f"prefetchFactor={prefetchFactor}"
        )
        # 训练模型
        epochNum = maxEpochs
        maxTrainWerBatches = 40

        if -1 != lastEpoch:
            epochN = epochNum - lastEpoch
        else:
            epochN = epochNum

        seed = 1
        for _ in range(epochN):
            moduleNet.train()
            freeze_backbone = moduleChoice == "LightTFNet" and hasattr(moduleNet, "conv2d") and epoch < freezeBackboneEpochs
            if moduleChoice == "LightTFNet" and hasattr(moduleNet, "conv2d"):
                for p in moduleNet.conv2d.parameters():
                    p.requires_grad = not freeze_backbone
                if freeze_backbone:
                    # 冻结阶段固定 BN/Dropout 统计，避免小 batch 下 backbone 漂移。
                    moduleNet.conv2d.eval()
            print(f"diagTrain backbone_frozen: {freeze_backbone}")

            scaler = GradScaler(enabled=(device.type == "cuda" and useAmp))
            loss_value = []
            trainWerScoreSum = 0.0
            trainWerSampleCount = 0
            trainPredLens = []
            trainLgtLens = []
            trainWerBatchCount = 0
            trainBlankProb = []
            invalidCtcBatchCount = 0
            invalidCtcSampleCount = 0
            optimizer.zero_grad(set_to_none=True)
            effective_train_batches = maxTrainBatches if maxTrainBatches > 0 else len(trainLoader)
            successful_backward_batches = 0
            for step_idx, Dict in enumerate(tqdm(stable(trainLoader, seed + epoch))):
                if step_idx >= effective_train_batches:
                    break
                try:
                    data = Dict["video"].to(device)
                    label = Dict["label"]
                    dataLen = Dict["videoLength"]
                    ##########################################################################
                    targetOutData = [yi.clone().detach().to(device) for yi in label]
                    targetData = targetOutData
                    targetLengths = torch.tensor(list(map(len, targetOutData)))
                    targetOutData = torch.cat(targetOutData, dim=0).to(device)

                    with autocast(enabled=(device.type == "cuda" and useAmp)):
                        logProbs1, logProbs2, logProbs3, logProbs4, logProbs5, lgt, x1, x2, x3 = moduleNet(data, dataLen, True)
                except Exception as e:
                    print("\n" + "="*50)
                    print("DETAILED ERROR IN TRAINING LOOP:")
                    import traceback
                    traceback.print_exc()
                    print("="*50)
                    raise RuntimeError(f"Training loop failed at epoch={epoch}, step={step_idx}") from e

                #########################################
                if "MSTNet" == moduleChoice:
                    logProbs1 = logSoftMax(logProbs1)
                    logProbs2 = logSoftMax(logProbs2)
                    logProbs3 = logSoftMax(logProbs3)
                    logProbs4 = logSoftMax(logProbs4)

                    loss1 = ctcLoss(logProbs1.float(), targetOutData, lgt, targetLengths)
                    loss2 = ctcLoss(logProbs2.float(), targetOutData, lgt, targetLengths)
                    loss3 = ctcLoss(logProbs3.float(), targetOutData, lgt * 2, targetLengths)
                    loss4 = ctcLoss(logProbs4.float(), targetOutData, lgt * 4, targetLengths)
                    loss = loss1 + loss2 + loss3 + loss4
                elif "LightTFNet" == moduleChoice:
                     logProbs1 = logSoftMax(logProbs1)
                     loss = ctcLoss(logProbs1.float(), targetOutData, lgt, targetLengths)
                elif "VAC" == moduleChoice or "CorrNet" == moduleChoice or "MAM-FSD" == moduleChoice \
                        or "SEN" == moduleChoice or "TFNet" == moduleChoice:
                    loss3 = 25 * kld(logProbs2, logProbs1, use_blank=False)

                    logProbs1 = logSoftMax(logProbs1)
                    logProbs2 = logSoftMax(logProbs2)

                    loss1 = ctcLoss(logProbs1, targetOutData, lgt, targetLengths).mean()
                    loss2 = ctcLoss(logProbs2, targetOutData, lgt, targetLengths).mean()
                    if "MAM-FSD" == moduleChoice:
                        loss4 = mseLoss(x1[0], x1[1])
                        loss5 = mseLoss(x2[0], x2[1])
                        loss6 = mseLoss(x3[0], x3[1])

                        loss = loss1 + loss2 + loss3 + 5 * loss4 + 1 * loss5 + 70 * loss6
                    elif "TFNet" == moduleChoice:
                        loss6 = 25 * kld(logProbs4, logProbs3, use_blank=False)

                        logProbs3 = logSoftMax(logProbs3)
                        logProbs4 = logSoftMax(logProbs4)

                        loss4 = ctcLoss(logProbs3, targetOutData, lgt, targetLengths).mean()
                        loss5 = ctcLoss(logProbs4, targetOutData, lgt, targetLengths).mean()

                        logProbs5 = logSoftMax(logProbs5)
                        loss7 = ctcLoss(logProbs5, targetOutData, lgt, targetLengths).mean()

                        loss = loss1 + loss2 + loss3 + loss4 + loss5 + loss6 + loss7
                    else:
                        loss = loss1 + loss2 + loss3

                if np.isinf(loss.item()) or np.isnan(loss.item()):
                    print('loss is nan')
                    continue

                if torch.is_tensor(lgt):
                    lgt_cpu = lgt.detach().view(-1).cpu().long()
                elif isinstance(lgt, list):
                    lgt_cpu = torch.tensor([
                        int(x.item()) if torch.is_tensor(x) else int(x) for x in lgt
                    ], dtype=torch.long)
                else:
                    lgt_cpu = torch.tensor([int(lgt)], dtype=torch.long)
                target_lengths_cpu = targetLengths.detach().view(-1).cpu().long()
                invalid_mask = lgt_cpu < target_lengths_cpu
                if invalid_mask.any():
                    invalidCtcBatchCount += 1
                    invalidCtcSampleCount += int(invalid_mask.sum().item())

                blank_prob_stats = _blank_prob_stats(logProbs1)
                trainBlankProb.append(blank_prob_stats["mean"])

                if trainWerBatchCount < maxTrainWerBatches:
                    with torch.no_grad():
                        pred_train, _ = decoder.decode(logProbs1.detach(), lgt, batch_first=False, probs=False)
                    pred_batch_indices = _pred_batch_to_indices(pred_train, word2idx)
                    for bi, pred_indices in enumerate(pred_batch_indices):
                        if bi < len(targetData):
                            trainWerScoreSum += WerScore([pred_indices], [targetData[bi]], idx2word, 1)
                            trainWerSampleCount += 1
                            trainPredLens.append(len(pred_indices))
                    if torch.is_tensor(lgt):
                        trainLgtLens.extend(lgt.detach().view(-1).cpu().tolist())
                    elif isinstance(lgt, list):
                        for x in lgt:
                            trainLgtLens.append(float(x.item()) if torch.is_tensor(x) else float(x))
                    trainWerBatchCount += 1

                scaler.scale(loss / gradAccumSteps).backward()
                successful_backward_batches += 1
                should_step = (successful_backward_batches % gradAccumSteps == 0)
                if should_step:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(moduleNet.parameters(), maxGradNorm)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)

                if "loss" in locals():
                    loss_value.append(loss.item())

                if device.type == "cuda" and ((step_idx + 1) % max(50, gradAccumSteps) == 0):
                    torch.cuda.empty_cache()

            if successful_backward_batches > 0 and (successful_backward_batches % gradAccumSteps != 0):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(moduleNet.parameters(), maxGradNorm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            print("epoch: %d, trainLoss: %.5f, lr: %f" % (
            epoch, np.mean(loss_value), optimizer.param_groups[0]['lr']))

            if trainWerSampleCount > 0:
                trainWer = trainWerScoreSum / trainWerSampleCount
                print(f"diagTrainWER(sampled): {trainWer:.2f} (samples={trainWerSampleCount}, batches={trainWerBatchCount})")
                print(f"diagTrain pred_len stats: {_format_stats(trainPredLens)}")
                print(f"diagTrain lgt stats: {_format_stats(trainLgtLens)}")
            if trainBlankProb:
                print(f"diagTrain blank_prob_mean: {np.mean(trainBlankProb):.6f}")
            if invalidCtcBatchCount > 0:
                print(
                    f"diagTrain invalid_ctc: batches={invalidCtcBatchCount}, "
                    f"samples={invalidCtcSampleCount}"
                )

            epoch = epoch + 1

            scheduler.step()

            moduleNet.eval()
            print("开始验证模型")
            # 验证模型
            werScoreSum = 0
            total_info = []
            total_sent = []
            loss_value = []
            validPredLens = []
            validLgtLens = []
            validBlankProb = []
            validWerSampleCount = 0
            effective_valid_batches = maxValidBatches if maxValidBatches > 0 else len(validLoader)
            for valid_idx, Dict in enumerate(tqdm(validLoader)):
                if valid_idx >= effective_valid_batches:
                    break
                data = Dict["video"].to(device)
                label = Dict["label"]
                dataLen = Dict["videoLength"]
                info = Dict["info"]
                ##########################################################################
                targetOutData = [yi.clone().detach().to(device) for yi in label]
                targetLengths = torch.tensor(list(map(len, targetOutData)))
                targetData = targetOutData
                targetOutData = torch.cat(targetOutData, dim=0).to(device)
                batchSize = len(targetLengths)

                with torch.no_grad():
                    logProbs1, logProbs2, logProbs3, logProbs4, logProbs5, lgt, x1, x2, x3 = moduleNet(data, dataLen, False)

                    logProbs1 = logSoftMax(logProbs1)
                    valid_blank_prob_stats = _blank_prob_stats(logProbs1)
                    validBlankProb.append(valid_blank_prob_stats["mean"])

                    if "MSTNet" == moduleChoice or "LightTFNet" == moduleChoice:
                        loss1 = ctcLoss(logProbs1, targetOutData, lgt, targetLengths)
                    else:
                        loss1 = ctcLoss(logProbs1, targetOutData, lgt, targetLengths).mean()

                    loss = loss1

                    if np.isinf(loss.item()) or np.isnan(loss.item()):
                        print('loss is nan')
                        continue

                loss_value.append(loss.item())
                ##########################################################################
                pred, targetOutDataCTC = decoder.decode(logProbs1, lgt, batch_first=False, probs=False)

                total_info += info
                total_sent += pred

                pred_batch_indices = _pred_batch_to_indices(pred, word2idx)
                if len(pred_batch_indices) < batchSize:
                    pred_batch_indices.extend([[] for _ in range(batchSize - len(pred_batch_indices))])

                if torch.is_tensor(lgt):
                    validLgtLens.extend(lgt.detach().view(-1).cpu().tolist())
                elif isinstance(lgt, list):
                    for x in lgt:
                        validLgtLens.append(float(x.item()) if torch.is_tensor(x) else float(x))

                for bi in range(batchSize):
                    pred_indices = pred_batch_indices[bi] if bi < len(pred_batch_indices) else []
                    validPredLens.append(len(pred_indices))

                    # Debug: compare one-sample reference and hypothesis for CE-CSL.
                    ref_indices = targetData[bi].tolist() if hasattr(targetData[bi], "tolist") else targetData[bi]
                    ref_sent = _indices_to_gloss(ref_indices, idx2word)
                    hyp_sent = _indices_to_gloss(pred_indices, idx2word)
                    if bi == 0:
                        raw_len = int(dataLen[bi].item()) if torch.is_tensor(dataLen[bi]) else int(dataLen[bi])
                        eff_len = int(lgt[bi].item()) if torch.is_tensor(lgt[bi]) else int(lgt[bi])
                        if logProbs1.dim() == 3:
                            sample_blank_prob = logProbs1[:, bi, 0].detach().exp().mean().item()
                        else:
                            sample_blank_prob = logProbs1[..., 0].detach().exp().mean().item()
                        print(f"\n[DEBUG Epoch {epoch}] Sample ID: {info[bi] if info else 'unknown'}")
                        print(f"Len: raw={raw_len}, lgt={eff_len}")
                        print(f"Ref: {ref_sent}")
                        print(f"Hyp: {hyp_sent} (pred_len={len(pred_indices)})")
                        print(f"blank_prob: {sample_blank_prob:.6f}")
                        if ref_sent.strip() != hyp_sent.strip():
                            print("--> MISMATCH detected!")

                    werScoreSum += WerScore([pred_indices], [targetData[bi]], idx2word, 1)
                    validWerSampleCount += 1
            if not os.path.exists('./wer/'):
                os.makedirs('./wer/')

            if device.type == "cuda":
                torch.cuda.empty_cache()

            currentLoss = np.mean(loss_value)

            werScore = werScoreSum / max(1, validWerSampleCount)

            if currentLoss < bestLoss:
                bestLoss = currentLoss
                bestLossEpoch = epoch - 1

            if werScore < bestWerScore:
                bestWerScore = werScore
                bestWerScoreEpoch = epoch - 1

                moduleDict = {}
                moduleDict['moduleNet_state_dict'] = moduleNet.state_dict()
                moduleDict['optimizer_state_dict'] = optimizer.state_dict()
                moduleDict['bestLoss'] = bestLoss
                moduleDict['bestLossEpoch'] = bestLossEpoch
                moduleDict['bestWerScore'] = bestWerScore
                moduleDict['bestWerScoreEpoch'] = bestWerScoreEpoch
                moduleDict['epoch'] = epoch
                torch.save(moduleDict, bestModuleSavePath)

            moduleDict = {}
            moduleDict['moduleNet_state_dict'] = moduleNet.state_dict()
            moduleDict['optimizer_state_dict'] = optimizer.state_dict()
            moduleDict['bestLoss'] = bestLoss
            moduleDict['bestLossEpoch'] = bestLossEpoch
            moduleDict['bestWerScore'] = bestWerScore
            moduleDict['bestWerScoreEpoch'] = bestWerScoreEpoch
            moduleDict['epoch'] = epoch
            torch.save(moduleDict, currentModuleSavePath)

            moduleSavePath1 = 'module/bestMoudleNet_' + str(epoch) + '.pth'
            torch.save(moduleDict, moduleSavePath1)

            # 保存识别结果到 wer 文件夹
            DataProcessMoudle.write2file('./wer/' + "output-hypothesis-{}{:0>4d}.ctm".format('dev', epoch), total_info, total_sent)

            print(f"validLoss: {currentLoss:.5f}, werScore: {werScore:.2f}")
            print(
                f"currentLoss: {currentLoss:.5f}, bestLoss: {bestLoss:.5f}, bestLossEpoch: {bestLossEpoch}, "
                f"bestWerScore: {bestWerScore:.2f}, bestWerScoreEpoch: {bestWerScoreEpoch}"
            )
            print(f"diagValid pred_len stats: {_format_stats(validPredLens)}")
            print(f"diagValid lgt stats: {_format_stats(validLgtLens)}")
            if validBlankProb:
                print(f"diagValid blank_prob_mean: {np.mean(validBlankProb):.6f}")
    else:
        bestWerScore = 65535
        offset = 1
        for i in range(55):
            currentModuleSavePath = "module/bestMoudleNet_" + str(i + offset) + ".pth"
            checkpoint = torch.load(currentModuleSavePath, map_location=torch.device('cpu'), weights_only=False)
            moduleNet.load_state_dict(checkpoint['moduleNet_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

            moduleNet.eval()
            print("开始验证模型")
            # 验证模型
            werScoreSum = 0
            loss_value = []
            total_info = []
            total_sent = []
            testWerSampleCount = 0

            if not os.path.exists('./wer/'):
                os.makedirs('./wer/')

            effective_test_batches = maxTestBatches if maxTestBatches > 0 else len(testLoader)
            for test_idx, Dict in enumerate(tqdm(testLoader)):
                if test_idx >= effective_test_batches:
                    break
                data = Dict["video"].to(device)
                label = Dict["label"]
                dataLen = Dict["videoLength"]
                info = Dict["info"]
                ##########################################################################
                targetOutData = [yi.clone().detach().to(device) for yi in label]
                targetLengths = torch.tensor(list(map(len, targetOutData)))
                targetData = targetOutData
                targetOutData = torch.cat(targetOutData, dim=0).to(device)
                batchSize = len(targetLengths)

                with torch.no_grad():
                    logProbs1, logProbs2, logProbs3, logProbs4, logProbs5, lgt, x1, x2, x3 = moduleNet(data, dataLen, False)

                    logProbs1 = logSoftMax(logProbs1)

                    loss1 = ctcLoss(logProbs1, targetOutData, lgt, targetLengths).mean()

                    loss = loss1

                loss_value.append(loss.item())

                pred, targetOutDataCTC = decoder.decode(logProbs1, lgt, batch_first=False, probs=False)

                total_info += info
                total_sent += pred

                pred_indices = _pred_to_indices(pred, word2idx)

                ref_indices = targetData[0].tolist() if hasattr(targetData[0], "tolist") else targetData[0]
                ref_sent = _indices_to_gloss(ref_indices, idx2word)
                hyp_sent = _indices_to_gloss(pred_indices, idx2word)
                print(f"\n[DEBUG TEST Epoch {i + offset}]")
                print(f"Ref: {ref_sent}")
                print(f"Hyp: {hyp_sent}")
                if ref_sent.strip() != hyp_sent.strip():
                    print("--> MISMATCH detected!")

                werScore = WerScore([pred_indices], targetData, idx2word, batchSize)
                werScoreSum = werScoreSum + werScore
                testWerSampleCount += batchSize

                if device.type == "cuda" and ((test_idx + 1) % 50 == 0):
                    torch.cuda.empty_cache()

            currentLoss = np.mean(loss_value)

            werScore = werScoreSum / max(1, testWerSampleCount)

            if werScore < bestWerScore:
                bestWerScore = werScore
                bestWerScoreEpoch = i + offset - 1

            bestLoss = currentLoss
            bestLossEpoch = i + offset - 1

            # 保存测试集识别结果
            DataProcessMoudle.write2file('./wer/' + "output-hypothesis-{}{:0>4d}.ctm".format('test', i+1), total_info, total_sent)

            print(f"testLoss: {currentLoss:.5f}, werScore: {werScore:.2f}")
            print(f"bestLoss: {bestLoss:.5f}, bestEpoch: {bestLossEpoch}, bestWerScore: {bestWerScore:.2f}, bestWerScoreEpoch: {bestWerScoreEpoch}")


