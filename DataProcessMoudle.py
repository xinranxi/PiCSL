#是数据处理模块：实现文本预处理（PreWords）、CE-CSL 词表构建（Word2Id）、
# 定义 CE-CSL 视频加载与标签映射的 PyTorch Dataset（MyDataset）、批次拼接/填充函数（collate_fn）
# 序列重塑与 CTC 解码/去空白工具（DataReshape、RemoveBlank、CTCGreedyDecode）
# 预测输出写文件（write2file）及一个用于序列级知识蒸馏的损失类（SeqKD）

# 支持数据集：CE-CSL、CSL。

# 目的：通过 PreWords / Word2Id 对不同数据集的标签做清洗与规范（去括号、统一标点/数字等）
# 生成词表并映射成 word2idx，供 MyDataset 加载视频帧与标签用于训练/CTC 解码。

import csv
import json
import os
import tempfile
import torch
from collections import defaultdict
from torch.utils.data import Dataset
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
import cv2

PAD = ' '


def _normalize_dataset_path(path_value):
    if not path_value:
        return None
    return os.path.normpath(str(path_value))


def _build_preprocessed_video_path(video_path, preprocessed_root, extension=".npy"):
    if not preprocessed_root:
        return None
    normalized_video_path = _normalize_dataset_path(video_path)
    normalized_root = _normalize_dataset_path(preprocessed_root)
    drive, tail = os.path.splitdrive(normalized_video_path)
    safe_tail = tail.lstrip("\\/")
    return os.path.join(normalized_root, safe_tail) + extension


def _build_cache_sidecar_path(cache_path):
    return cache_path + ".json"


def _safe_int(value, default):
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _detect_split_name(path_value):
    normalized = str(path_value).replace("\\", "/").lower()
    if "train" in normalized:
        return "train"
    if "valid" in normalized or "dev" in normalized:
        return "valid"
    if "test" in normalized:
        return "test"
    return "unknown"


def _atomic_save_compressed_array(cache_path, frames, meta):
    cache_dir = os.path.dirname(cache_path)
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)

    fd, tmp_path = tempfile.mkstemp(prefix="cache_", suffix=".npz", dir=cache_dir or None)
    os.close(fd)
    try:
        np.savez_compressed(tmp_path, frames=frames)
        os.replace(tmp_path, cache_path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    with open(_build_cache_sidecar_path(cache_path), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


def _load_cached_frames(cache_path):
    if cache_path.endswith(".npy"):
        frames = np.load(cache_path)
    else:
        with np.load(cache_path) as data:
            frames = data["frames"]

    if frames.ndim != 4:
        raise ValueError(f"Invalid cached frames shape for {cache_path}: {frames.shape}")
    return frames


def _read_split_manifest(manifest_path):
    samples = []
    with open(manifest_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split('\t')
            if len(parts) < 2:
                parts = line.split()
            if len(parts) < 2:
                continue
            samples.append((parts[0], parts[1]))
    return samples


def _read_csl_corpus(label_path):
    label_map = {}
    with open(label_path, 'r', encoding='utf-8') as f:
        for raw_line in f:
            line = raw_line.replace('\ufeff', '').strip()
            if not line:
                continue
            parts = line.split(None, 1)
            if len(parts) < 2:
                continue
            label_map[parts[0].strip()] = parts[1].strip().replace('\ufeff', '')
    return label_map


def _collect_csl_active_ids(*paths):
    active_ids = []
    for path in paths:
        if not path:
            continue
        if os.path.isfile(path):
            for _, label_key in _read_split_manifest(path):
                key = str(label_key).strip().replace('\ufeff', '')
                if key:
                    active_ids.append(key)
        elif os.path.isdir(path):
            for name in sorted(os.listdir(path)):
                folder = str(name).strip().replace('\ufeff', '')
                if folder and os.path.isdir(os.path.join(path, name)):
                    active_ids.append(folder)
    # 保持顺序去重，便于调试 vocab 来源。
    return list(dict.fromkeys(active_ids))

def PreWords(words):# 预处理文本
    for i in range(len(words)):
        word = str(words[i]).strip()

        if not word:
            words[i] = word
            continue

        n = 0
        subFlag = False
        wordList = list(word)
        for j in range(len(word)):
            if word[j] == "(" or word[j] == "{" or word[j] == "[" or word[j] == "（":
                subFlag = True

            if subFlag:
                wordList.pop(j - n)
                n = n + 1

            if word[j] == ")" or word[j] == "}" or word[j] == "]" or word[j] == "）":
                subFlag = False

        word = "".join(wordList).strip()

        if not word:
            words[i] = word
            continue

        if word[-1].isdigit():
            if not word[0].isdigit():
                wordList = list(word)
                wordList.pop(len(word) - 1)
                word = "".join(wordList).strip()

        if not word:
            words[i] = word
            continue

        if word[0] == "," or word[0] == "，":
            wordList = list(word)
            wordList[0] = '，'
            word = ''.join(wordList)

        if word[0] == "?" or word[0] == "？":
            wordList = list(word)
            wordList[0] = '？'
            word = ''.join(wordList)

        if word.isdigit():
            word = str(int(word))

        words[i] = word

    return words

def Word2Id(trainLabelPath, validLabelPath, testLabelPath, dataSetName,
            trainDataPath=None, validDataPath=None, testDataPath=None):
    if dataSetName not in ("CE-CSL", "CSL"):
        raise ValueError(f"Unsupported dataset: {dataSetName}. Supported: CE-CSL, CSL")

    wordList = []

    if dataSetName == "CE-CSL":
        for label_path in [trainLabelPath, validLabelPath, testLabelPath]:
            with open(label_path, 'r', encoding="utf-8") as f:
                reader = csv.reader(f)
                for n, row in enumerate(reader):
                    if n == 0:
                        continue
                    # Robustly find Gloss column by looking for '/' separator.
                    gloss_idx = 3
                    max_slashes = 0
                    for i, col in enumerate(row):
                        slashes = col.count('/')
                        if slashes > max_slashes:
                            max_slashes = slashes
                            gloss_idx = i

                    words = row[gloss_idx].split("/")
                    words = [w for w in PreWords(words) if w]
                    wordList += words

    elif dataSetName == "CSL":
        # CSL: 词表只基于当前 split 实际会训练/验证/测试到的句子构建，
        # 避免把整份 corpus 的大量无关字符加入输出空间，降低 CTC blank 塌缩风险。
        seen_paths = set()
        csl_label_map = {}
        for label_path in [trainLabelPath, validLabelPath, testLabelPath]:
            if label_path in seen_paths:
                continue
            seen_paths.add(label_path)
            csl_label_map.update(_read_csl_corpus(label_path))

        active_ids = _collect_csl_active_ids(trainDataPath, validDataPath, testDataPath)
        selected_ids = [video_id for video_id in active_ids if video_id in csl_label_map]
        if not selected_ids:
            selected_ids = sorted(csl_label_map.keys())

        for video_id in selected_ids:
            sentence = csl_label_map[video_id]
            for char in sentence:
                if char.strip():
                    wordList.append(char)

    idx2word = [PAD]
    set2list = sorted(list(set(wordList)))
    idx2word.extend(set2list)

    word2idx = {w: i for i, w in enumerate(idx2word)}

    return word2idx, len(idx2word) - 1, idx2word


class MyDataset(Dataset):
    def __init__(self, ImagePath, LabelPath, word2idx, dataSetName, isTrain=False, transform=None, frameSampleStride=1,
                 preprocessedRoot=None, usePreprocessed=0, videoCacheMode="off", videoCacheFormat="npz",
                 cacheTrainOnly=1, cacheInMemoryItems=0):
        """
        path : 数据路径，包含了图像的路径
        transform：数据处理，对图像进行随机剪裁，以及转换成tensor
        """
        self.ImagePath = ImagePath
        self.transform = transform
        self.dataSetName = dataSetName
        self.p_drop = 0.5
        self.random_drop = True
        self.isTrain = isTrain
        self.frameSampleStride = max(1, int(frameSampleStride))
        self.preprocessedRoot = _normalize_dataset_path(preprocessedRoot)
        self.usePreprocessed = bool(int(usePreprocessed))
        self.videoCacheMode = str(videoCacheMode or "off").strip().lower()
        self.videoCacheFormat = str(videoCacheFormat or "npz").strip().lower()
        self.cacheTrainOnly = bool(int(cacheTrainOnly))
        self.cacheInMemoryItems = max(0, _safe_int(cacheInMemoryItems, 0))
        self.splitName = _detect_split_name(ImagePath)
        self._memory_cache = {}
        self._memory_cache_order = []
        self.cacheHits = 0
        self.cacheMisses = 0
        self.cacheWrites = 0
        self.rawReads = 0

        if dataSetName not in ("CE-CSL", "CSL"):
            raise ValueError(f"Unsupported dataset: {dataSetName}. Supported: CE-CSL, CSL")
        if self.videoCacheFormat not in ("npz", "npy"):
            raise ValueError(f"Unsupported cache format: {self.videoCacheFormat}. Supported: npz, npy")

        lable = {}

        if dataSetName == "CE-CSL":
            lableDict = {}
            with open(LabelPath, 'r', encoding="utf-8") as f:
                reader = csv.reader(f)
                for n, row in enumerate(reader):
                    if n == 0:
                        continue
                    # Robustly find Gloss column by looking for '/' separator.
                    gloss_idx = 3
                    max_slashes = 0
                    for i, col in enumerate(row):
                        slashes = col.count('/')
                        if slashes > max_slashes:
                            max_slashes = slashes
                            gloss_idx = i

                    if len(row) > 0:
                        lableDict[row[0]] = row[gloss_idx]

            for line in lableDict:
                sentences = lableDict[line].split("/")
                sentences = [w for w in PreWords(sentences) if w]

                txtInt = []
                for i in sentences:
                    txtInt.append(word2idx[i])

                lable[line] = txtInt

        elif dataSetName == "CSL":
            # CSL: LabelPath = corpus.txt，格式: "{id} {句子}"
            csl_label_map = _read_csl_corpus(LabelPath)
            for video_id, sentence in csl_label_map.items():
                txtInt = []
                for char in sentence:
                    if char.strip() and char in word2idx:
                        txtInt.append(word2idx[char])
                lable[video_id] = txtInt

        imgs = []

        if dataSetName == "CSL" and os.path.isfile(ImagePath):
            manifest_samples = _read_split_manifest(ImagePath)
            for rel_video_path, label_key in manifest_samples:
                normalized_video_path = os.path.normpath(rel_video_path)
                if not os.path.isabs(normalized_video_path):
                    normalized_video_path = os.path.normpath(os.path.join(os.getcwd(), normalized_video_path))
                if label_key in lable:
                    imgs.append((normalized_video_path, lable[label_key]))
                else:
                    print(f"Warning: No label for CSL manifest key {label_key}")
        else:
            # 扫描视频文件
            fileNames = sorted(os.listdir(ImagePath))
            for name in fileNames:
                dirPath = os.path.join(ImagePath, name)
                if not os.path.isdir(dirPath):
                    continue
                videoFiles = sorted(os.listdir(dirPath))
                for videoFile in videoFiles:
                    videoName = os.path.splitext(videoFile)[0]
                    if dataSetName == "CSL":
                        # CSL 视频命名: P01_s1_00_0_color.avi，所在文件夹名 = video_id
                        # 用文件夹名作为标签 key
                        if name in lable:
                            videoPath = os.path.join(dirPath, videoFile)
                            imgs.append((videoPath, lable[name]))
                        else:
                            print(f"Warning: No label for CSL folder {name}")
                    else:
                        # CE-CSL: 视频文件名（不含扩展名） = label key
                        if videoName in lable:
                            videoPath = os.path.join(dirPath, videoFile)
                            imgs.append((videoPath, lable[videoName]))
                        else:
                            print(f"Warning: No label for video {videoName}")

        self.imgs = imgs

    def _is_cache_enabled_for_split(self):
        if not self.preprocessedRoot:
            return False
        if self.videoCacheMode == "off" and not self.usePreprocessed:
            return False
        if self.cacheTrainOnly and self.splitName != "train":
            return False
        return True

    def _get_cache_path_candidates(self, video_path):
        preferred_ext = ".npz" if self.videoCacheFormat == "npz" else ".npy"
        candidates = []
        preferred = _build_preprocessed_video_path(video_path, self.preprocessedRoot, preferred_ext)
        if preferred:
            candidates.append(preferred)
        legacy_ext = ".npy" if preferred_ext == ".npz" else ".npz"
        legacy = _build_preprocessed_video_path(video_path, self.preprocessedRoot, legacy_ext)
        if legacy and legacy not in candidates:
            candidates.append(legacy)
        return candidates

    def _memory_get(self, key):
        if self.cacheInMemoryItems <= 0:
            return None
        value = self._memory_cache.get(key)
        if value is None:
            return None
        if key in self._memory_cache_order:
            self._memory_cache_order.remove(key)
        self._memory_cache_order.append(key)
        return value

    def _memory_put(self, key, value):
        if self.cacheInMemoryItems <= 0:
            return
        if key in self._memory_cache_order:
            self._memory_cache_order.remove(key)
        self._memory_cache[key] = value
        self._memory_cache_order.append(key)
        while len(self._memory_cache_order) > self.cacheInMemoryItems:
            old_key = self._memory_cache_order.pop(0)
            self._memory_cache.pop(old_key, None)

    def sample_indices(self, n):
        stride = self.frameSampleStride
        indices = np.arange(0, n, stride, dtype=int)
        # 确保最后一帧被采样，避免截断末尾动作信息。
        if len(indices) == 0:
            indices = np.array([0], dtype=int)
        elif indices[-1] != n - 1:
            indices = np.append(indices, n - 1)
        return indices

    def _read_video_frames(self, fn):
        cap = cv2.VideoCapture(fn)
        frames = []
        stride = self.frameSampleStride
        frame_idx = 0
        last_kept_frame = None

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            keep = (frame_idx % stride == 0)
            if keep:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (224, 224))
                frames.append(frame)
                last_kept_frame = frame

            frame_idx += 1

        cap.release()

        # 旧实现会重新打开视频并 seek 到最后一帧；某些 AVI 在这里会极慢甚至假死。
        # 这里改为直接复用最后一次成功保留的帧，避免末帧随机访问导致训练卡住。
        if frame_idx > 0 and (frame_idx - 1) % stride != 0 and last_kept_frame is not None:
            frames.append(last_kept_frame.copy())

        return frames

    def _read_preprocessed_frames(self, video_path):
        cache_key = os.path.normpath(video_path)
        memory_frames = self._memory_get(cache_key)
        if memory_frames is not None:
            self.cacheHits += 1
            return memory_frames

        for preprocessed_path in self._get_cache_path_candidates(video_path):
            if preprocessed_path is None or not os.path.exists(preprocessed_path):
                continue
            frames = _load_cached_frames(preprocessed_path)
            self._memory_put(cache_key, frames)
            self.cacheHits += 1
            return frames

        self.cacheMisses += 1
        return None

    def _write_preprocessed_frames(self, video_path, frames):
        if not self._is_cache_enabled_for_split():
            return
        if self.videoCacheMode != "lazy":
            return

        cache_ext = ".npz" if self.videoCacheFormat == "npz" else ".npy"
        cache_path = _build_preprocessed_video_path(video_path, self.preprocessedRoot, cache_ext)
        if cache_path is None or os.path.exists(cache_path):
            return

        meta = {
            "video_path": os.path.normpath(video_path),
            "split": self.splitName,
            "frame_sample_stride": self.frameSampleStride,
            "shape": list(frames.shape),
            "dtype": str(frames.dtype),
            "cache_format": self.videoCacheFormat,
        }
        cache_dir = os.path.dirname(cache_path)
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)

        if self.videoCacheFormat == "npy":
            np.save(cache_path, frames)
            with open(_build_cache_sidecar_path(cache_path), "w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)
        else:
            _atomic_save_compressed_array(cache_path, frames, meta)
        self.cacheWrites += 1
        self._memory_put(os.path.normpath(video_path), frames)

    def __getitem__(self, index):
        fn, label = self.imgs[index]# 通过index索引返回一个图像路径fn 与 标签label
        # CE-CSL (.mp4) 和 CSL (.avi) 都用 cv2.VideoCapture 读取，逻辑一致

        # fn 是视频文件路径，如 CE-CSL/train/A/train-00001.mp4
        info = os.path.basename(fn)

        frames = None
        if self.usePreprocessed or self.videoCacheMode in ("readonly", "lazy"):
            frames = self._read_preprocessed_frames(fn)

        if frames is None:
            frames = self._read_video_frames(fn)
            self.rawReads += 1
            frames = np.asarray(frames, dtype=np.uint8)
            self._write_preprocessed_frames(fn, frames)

        if len(frames) == 0:
            print(f"Warning: Video {fn} has 0 frames!")
            return {"video": torch.zeros((1, 3, 224, 224)), "label": label, "info": info}

        imgSeq = self.transform(frames)
        if isinstance(imgSeq, torch.Tensor):
            imgSeq = imgSeq.float() / 127.5 - 1
        else:
            imgSeq = [torch.from_numpy(f).float() / 127.5 - 1 for f in imgSeq]

        sample = {"video": imgSeq, "label": label, "info": info}

        return sample  # 这就返回一个样本

    def __len__(self):
        return len(self.imgs)  # 返回长度，index就会自动的指导读取多少

class defaultdict_with_warning(defaultdict):
    warned = set()
    warning_enabled = False

    def __getitem__(self, key):
        if key == "text" and key not in self.warned and self.warning_enabled:
            print(
                'Warning: using batch["text"] to obtain label is deprecated, '
                'please use batch["label"] instead.'
            )
            self.warned.add(key)

        return super().__getitem__(key)

def collate_fn(batch):
    collated = defaultdict_with_warning(list)

    batch = [item for item in sorted(batch, key=lambda x: len(x["video"]), reverse=True)]
    max_video_len = len(batch[0]["video"])

    # MAM-FSD、CorrNet、VAC、TFNet
    left_pad = 6# 6
    total_stride = 4# 4
    right_pad = int(np.ceil(max_video_len / total_stride)) * total_stride - max_video_len + left_pad
    padded_max_len = max_video_len + left_pad + right_pad

    # MSTNet
    # left_pad = 0  # 6
    # total_stride = 4  # 4
    # right_pad = int(np.ceil(max_len / total_stride)) * total_stride - max_len + left_pad
    # max_len = max_len + left_pad + right_pad

    padded_video = []
    for sample in batch:
        vid = sample["video"]
        collated["videoLength"].append(torch.LongTensor([len(vid)]))
        padded_video.append(torch.cat(
            (
                vid[0][None].expand(left_pad, -1, -1, -1),
                vid,
                vid[-1][None].expand(padded_max_len - len(vid) - left_pad, -1, -1, -1),
            )
            , dim=0))

        collated["label"].append(torch.tensor(sample["label"]).long())
        collated["info"].append(sample["info"])
        collated["expand"].append([left_pad, padded_max_len - len(vid) - left_pad])

    padded_video = torch.stack(padded_video)
    collated["video"] = padded_video
    collated.warning_enabled = True

    return dict(collated)

def DataReshape(seqData, device):
    xl = list(map(len, seqData))
    batchSize = len(xl)
    seqData = torch.cat(seqData, dim=0).to(device)

    return seqData, batchSize, xl

def RemoveBlank(labels, maxSentenceLen, blank=0):
    new_labels = []
    # 合并相同的标签
    previous = None
    for l in labels:
        if l != previous:
            new_labels.append(l)
            previous = l
    # 删除blank
    new_labels = [l for l in new_labels if l != blank]

    if len(new_labels) < maxSentenceLen:
        for _ in range(maxSentenceLen - len(new_labels)):
            new_labels.append(0)
        new_labelsTmp = new_labels
    else:
        new_labelsTmp = new_labels[:maxSentenceLen]

    outPut = torch.Tensor(new_labelsTmp)

    return outPut

def CTCGreedyDecode(y, maxSentenceLen, blank=0):
    # 按列取最大值，即每个时刻t上最大值对应的下标
    raw_rs = y.argmax(dim=-1)
    # 移除blank,值为0的位置表示这个位置是blank
    rs = RemoveBlank(raw_rs, maxSentenceLen, blank)
    return rs

def write2file(path, info, output):
    filereader = open(path, "w")
    for sample_idx, sample in enumerate(output):
        for word_idx, word in enumerate(sample):
            filereader.writelines(
                "{} 1 {:.2f} {:.2f} {}\n".format(info[sample_idx],
                                                 word_idx * 1.0 / 100,
                                                 (word_idx + 1) * 1.0 / 100,
                                                 word[0]))

class SeqKD(nn.Module):
    """
    NLL loss with label smoothing.
    """

    def __init__(self, T=1):
        super(SeqKD, self).__init__()
        self.kdloss = nn.KLDivLoss(reduction='batchmean')
        self.T = T

    def forward(self, prediction_logits, ref_logits, use_blank=True):
        start_idx = 0 if use_blank else 1
        prediction_logits = F.log_softmax(prediction_logits[:, :, start_idx:]/self.T, dim=-1) \
            .view(-1, ref_logits.shape[2] - start_idx)
        ref_probs = F.softmax(ref_logits[:, :, start_idx:]/self.T, dim=-1) \
            .view(-1, ref_logits.shape[2] - start_idx)
        loss = self.kdloss(prediction_logits, ref_probs)*self.T*self.T
        return loss
