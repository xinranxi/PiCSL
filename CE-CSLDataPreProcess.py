import random
import os
import numpy as np
from tqdm import tqdm
import cv2

def seed_torch(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

def main(dataPath, saveDataPath):
    fileTypes = sorted(os.listdir(dataPath))

    framesList = []
    fpsList = []
    videoTimeList = []
    resolutionList = []
    
    for fileType in fileTypes:
        filePath = os.path.join(dataPath, fileType)
        # 只处理文件夹，跳过 .csv 等文件
        if not os.path.isdir(filePath) or fileType == "frames":
            continue
            
        saveFilePath = os.path.join(saveDataPath, fileType)
        translators = sorted(os.listdir(filePath))

        for translator in translators:
            translatorPath = os.path.join(filePath, translator)
            if not os.path.isdir(translatorPath):
                continue
                
            saveTranslatorPath = os.path.join(saveFilePath, translator)
            videos = sorted(os.listdir(translatorPath))

            for video in tqdm(videos, desc=f"Processing {fileType}/{translator}"):
                if not video.endswith('.mp4'):
                    continue
                    
                videoPath = os.path.join(translatorPath, video)
                nameString = video.split(".")[0]
                saveImagePath = os.path.join(saveTranslatorPath, nameString)

                if not os.path.exists(saveImagePath):
                    os.makedirs(saveImagePath)

                try:
                    cap = cv2.VideoCapture(videoPath)
                    nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
                    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
                    resolution = (width, height)
                    videoTime = nframes / fps if fps > 0 else 0

                    framesList.append(nframes)
                    fpsList.append(fps)
                    videoTimeList.append(videoTime)
                    resolutionList.append(resolution)

                    for i in range(nframes):
                        ret, image = cap.read()
                        if not ret:
                            break
                        try:
                            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                            image = cv2.resize(image, (256, 256))

                            nameString_idx = str(i).zfill(5)
                            imagePath = os.path.join(saveImagePath, f"{nameString_idx}.jpg")
                            # 使用 imencode 避免中文或特殊路径报错
                            cv2.imencode('.jpg', cv2.cvtColor(image, cv2.COLOR_RGB2BGR))[1].tofile(imagePath)
                        except Exception as e:
                            pass
                    cap.release()
                except Exception as e:
                    print(f"Error processing {videoPath}: {e}")

    if framesList:
        print(f"Max Frames Number:{max(framesList)}\n"
              f"Min Frames Number:{min(framesList)}\n"
              f"Max Video Time:{max(videoTimeList)}\n"
              f"Min Video Time:{min(videoTimeList)}\n"
              f"Fps Set:{set(fpsList)}\n"
              f"Resolution Set:{set(resolutionList)}\n")
    else:
        print("No valid mp4 videos found.")

if __name__ == '__main__':
    dataPath = r"D:\Graduation_Thesis\TEST\CE-CSL"
    saveDataPath = r"D:\Graduation_Thesis\TEST\CE-CSL\frames"

    seed_torch()
    main(dataPath, saveDataPath)
