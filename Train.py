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
    if dataSetName != "CE-CSL":
        raise ValueError(f"This project currently supports CE-CSL only, got: {dataSetName}")
    gradAccumSteps = max(1, int(configParams.get("gradAccumSteps", 8)))
    maxGradNorm = float(configParams.get("maxGradNorm", 5.0))
    freezeBackboneEpochs = max(0, int(configParams.get("freezeBackboneEpochs", 3)))
    backboneLrScale = float(configParams.get("backboneLrScale", 0.1))
    max_num_states = 1

    # 预处理语言序列
    word2idx, wordSetNum, idx2word = DataProcessMoudle.Word2Id(trainLabelPath, validLabelPath, testLabelPath, dataSetName)
    # 图像预处理：与 TFNet-main 对齐，保留翻转和颜色抖动。
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
    trainData = DataProcessMoudle.MyDataset(trainDataPath, trainLabelPath, word2idx, dataSetName, isTrain=True, transform=transform)

    validData = DataProcessMoudle.MyDataset(validDataPath, validLabelPath, word2idx, dataSetName, transform=transformTest)

    testData = DataProcessMoudle.MyDataset(testDataPath, testLabelPath, word2idx, dataSetName, transform=transformTest)

    trainLoader = DataLoader(dataset=trainData, batch_size=batchSize, shuffle=True, num_workers=numWorkers,
                             pin_memory=pinmMemory, collate_fn=DataProcessMoudle.collate_fn, drop_last=True)
    validLoader = DataLoader(dataset=validData, batch_size=1, shuffle=False, num_workers=numWorkers,
                             pin_memory=pinmMemory, collate_fn=DataProcessMoudle.collate_fn, drop_last=True)
    testLoader = DataLoader(dataset=testData, batch_size=1, shuffle=False, num_workers=numWorkers,
                            pin_memory=pinmMemory, collate_fn=DataProcessMoudle.collate_fn, drop_last=True)

    # 定义模型
    moduleNet = Net.moduleNet(hiddenSize, wordSetNum * max_num_states + 1, moduleChoice, device, dataSetName, True)
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
        checkpoint = torch.load(currentModuleSavePath, map_location=torch.device('cpu'))
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
        # 训练模型
        epochNum = 55
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

            scaler = GradScaler()
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
            for step_idx, Dict in enumerate(tqdm(stable(trainLoader, seed + epoch))):
                try:
                    data = Dict["video"].to(device)
                    label = Dict["label"]
                    dataLen = Dict["videoLength"]
                    ##########################################################################
                    targetOutData = [yi.clone().detach().to(device) for yi in label]
                    targetData = targetOutData
                    targetLengths = torch.tensor(list(map(len, targetOutData)))
                    targetOutData = torch.cat(targetOutData, dim=0).to(device)

                    with autocast():
                        logProbs1, logProbs2, logProbs3, logProbs4, logProbs5, lgt, x1, x2, x3 = moduleNet(data, dataLen, True)
                except Exception as e:
                    print("\n" + "="*50)
                    print("DETAILED ERROR IN TRAINING LOOP:")
                    import traceback
                    traceback.print_exc()
                    print("="*50)
                    exit(1)

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

                blank_prob = logProbs1.detach().exp()[..., 0].mean().item()
                trainBlankProb.append(blank_prob)

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
                should_step = ((step_idx + 1) % gradAccumSteps == 0) or ((step_idx + 1) == len(trainLoader))
                if should_step:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(moduleNet.parameters(), maxGradNorm)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)

                if "loss" in locals():
                    loss_value.append(loss.item())

                torch.cuda.empty_cache()

            print("epoch: %d, trainLoss: %.5f, lr: %f" % (
            epoch, np.mean(loss_value), optimizer.param_groups[0]['lr']))

            if trainWerSampleCount > 0:
                trainWer = trainWerScoreSum / trainWerSampleCount
                print(f"diagTrainWER(sampled): {trainWer:.2f} (samples={trainWerSampleCount}, batches={trainWerBatchCount})")
                print(f"diagTrain pred_len stats: {_format_stats(trainPredLens)}")
                print(f"diagTrain lgt stats: {_format_stats(trainLgtLens)}")
            if trainBlankProb:
                print(f"diagTrain blank_prob_mean: {np.mean(trainBlankProb):.4f}")
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
            for Dict in tqdm(validLoader):
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
                    validBlankProb.append(logProbs1.exp()[..., 0].mean().item())

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

                pred_indices = _pred_to_indices(pred, word2idx)
                validPredLens.append(len(pred_indices))

                # Debug: compare one-sample reference and hypothesis for CE-CSL.
                ref_indices = targetData[0].tolist() if hasattr(targetData[0], "tolist") else targetData[0]
                ref_sent = _indices_to_gloss(ref_indices, idx2word)
                hyp_sent = _indices_to_gloss(pred_indices, idx2word)
                raw_len = int(dataLen[0].item()) if torch.is_tensor(dataLen[0]) else int(dataLen[0])
                eff_len = int(lgt[0].item()) if torch.is_tensor(lgt[0]) else int(lgt[0])
                validLgtLens.append(eff_len)
                print(f"\n[DEBUG Epoch {epoch}] Sample ID: {info[0] if info else 'unknown'}")
                print(f"Len: raw={raw_len}, lgt={eff_len}")
                print(f"Ref: {ref_sent}")
                print(f"Hyp: {hyp_sent} (pred_len={len(pred_indices)})")
                if ref_sent.strip() != hyp_sent.strip():
                    print("--> MISMATCH detected!")

                werScore = WerScore([pred_indices], targetData, idx2word, batchSize)
                werScoreSum = werScoreSum + werScore
            if not os.path.exists('./wer/'):
                os.makedirs('./wer/')

            torch.cuda.empty_cache()

            currentLoss = np.mean(loss_value)

            werScore = werScoreSum / len(validLoader)

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

            bestLoss = currentLoss
            bestLossEpoch = epoch - 1

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
            print(f"bestLoss: {bestLoss:.5f}, beatEpoch: {bestLossEpoch}, bestWerScore: {bestWerScore:.2f}, bestWerScoreEpoch: {bestWerScoreEpoch}")
            print(f"diagValid pred_len stats: {_format_stats(validPredLens)}")
            print(f"diagValid lgt stats: {_format_stats(validLgtLens)}")
            if validBlankProb:
                print(f"diagValid blank_prob_mean: {np.mean(validBlankProb):.4f}")
    else:
        bestWerScore = 65535
        offset = 1
        for i in range(55):
            currentModuleSavePath = "module/bestMoudleNet_" + str(i + offset) + ".pth"
            checkpoint = torch.load(currentModuleSavePath, map_location=torch.device('cpu'))
            moduleNet.load_state_dict(checkpoint['moduleNet_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

            moduleNet.eval()
            print("开始验证模型")
            # 验证模型
            werScoreSum = 0
            loss_value = []
            total_info = []
            total_sent = []

            if not os.path.exists('./wer/'):
                os.makedirs('./wer/')

            for Dict in tqdm(testLoader):
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

                torch.cuda.empty_cache()

            currentLoss = np.mean(loss_value)

            werScore = werScoreSum / len(testLoader)

            if werScore < bestWerScore:
                bestWerScore = werScore
                bestWerScoreEpoch = i + offset - 1

            bestLoss = currentLoss
            bestLossEpoch = i + offset - 1

            # 保存测试集识别结果
            DataProcessMoudle.write2file('./wer/' + "output-hypothesis-{}{:0>4d}.ctm".format('test', i+1), total_info, total_sent)

            print(f"testLoss: {currentLoss:.5f}, werScore: {werScore:.2f}")
            print(f"bestLoss: {bestLoss:.5f}, bestEpoch: {bestLossEpoch}, bestWerScore: {bestWerScore:.2f}, bestWerScoreEpoch: {bestWerScoreEpoch}")


