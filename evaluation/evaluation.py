import os
import shutil
import subprocess
import sys
import re
from evaluation.python_wer_evaluation import wer_calculation
from evaluation.python_wer_evaluation import wer_calculation1
from evaluation.python_wer_evaluation import wer_calculation2

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def _run_command(command, cwd=None, output_path=None):
    with open(output_path, "w", encoding="utf-8") if output_path else open(os.devnull, "w", encoding="utf-8") as fh:
        subprocess.run(command, cwd=cwd or BASE_DIR, check=True, stdout=fh)


def _prepare_ctm(output_file, tmp_ctm, tmp2_ctm):
    preprocess_script = os.path.join(BASE_DIR, "preprocess.sh")
    git_bash = shutil.which("bash")
    if git_bash:
        subprocess.run([git_bash, preprocess_script, os.path.join(BASE_DIR, output_file), tmp_ctm, tmp2_ctm], cwd=BASE_DIR, check=True)
    else:
        _preprocess_ctm_python(os.path.join(BASE_DIR, output_file), tmp2_ctm)


def _sort_stm(source_path, target_path):
    with open(source_path, "r", encoding="utf-8") as fr:
        lines = sorted(fr.readlines(), key=lambda line: line.split()[0] if line.split() else "")
    with open(target_path, "w", encoding="utf-8") as fw:
        fw.writelines(lines)


def _collapse_letter_tokens(text):
    pattern = re.compile(r'(?<![\w-])([A-Z]+)(?:\s+\1)+(?![\w-])')
    prev = None
    while prev != text:
        prev = text
        text = pattern.sub(lambda m: m.group(1), text)

    token_pattern = re.compile(r'(?<!\S)([A-Z]|SCH|NN)(?:\s+([A-Z]|NN))(?!\S)')
    prev = None
    while prev != text:
        prev = text
        text = token_pattern.sub(lambda m: f"{m.group(1)}+{m.group(2)}", text)
    return text


def _normalize_ctm_token_text(text):
    text = text.replace("loc-", "")
    text = text.replace("cl-", "")
    text = text.replace("qu-", "")
    text = text.replace("poss-", "")
    text = text.replace("lh-", "")
    text = text.replace("S0NNE", "SONNE")
    text = text.replace("HABEN2", "HABEN")
    text = text.replace("__EMOTION__", "")
    text = text.replace("__PU__", "")
    text = text.replace("__LEFTHAND__", "")
    text = text.replace("WIE AUSSEHEN", "WIE-AUSSEHEN")
    text = text.replace("ZEIGEN ", "ZEIGEN-BILDSCHIRM ")
    if text.endswith("ZEIGEN"):
        text = text[:-6] + "ZEIGEN-BILDSCHIRM"
    text = text.replace("-PLUSPLUS", "")
    text = re.sub(r'\b([A-Z][A-Z]*)RAUM\b', r'\1', text)
    text = _collapse_letter_tokens(text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def _preprocess_ctm_python(source_path, output_path):
    processed_rows = []
    with open(source_path, "r", encoding="utf-8") as fr:
        for raw in fr:
            parts = raw.strip().split()
            if len(parts) < 5:
                continue
            token_text = " ".join(parts[4:])
            token_text = _normalize_ctm_token_text(token_text)
            if any(tag in token_text for tag in ["__LEFTHAND__", "__EPENTHESIS__", "__EMOTION__"]):
                continue
            if token_text:
                processed_rows.append(parts[:4] + [token_text])

    final_rows = []
    last_id = None
    last_row = None
    counts = {}
    for row in sorted(processed_rows, key=lambda item: (item[0], float(item[2]))):
        row_id = row[0]
        if last_id != row_id and last_id is not None and counts.get(last_id, 0) < 1 and last_row is not None:
            final_rows.append(last_row[:4] + ["[EMPTY]"])
        if row[4]:
            counts[row_id] = counts.get(row_id, 0) + 1
            final_rows.append(row)
        last_id = row_id
        last_row = row
    if last_id is not None and counts.get(last_id, 0) < 1 and last_row is not None:
        final_rows.append(last_row[:4] + ["[EMPTY]"])

    with open(output_path, "w", encoding="utf-8") as fw:
        for row in final_rows:
            fw.write(" ".join(row) + "\n")

def evaluate3(mode="dev", evaluate_prefix=None,
             output_file=None, isPrint=True):
    output_path = os.path.join(BASE_DIR, output_file)
    tmp_ctm = os.path.join(BASE_DIR, "tmp.ctm")
    tmp2_ctm = os.path.join(BASE_DIR, "tmp2.ctm")
    tmp_stm = os.path.join(BASE_DIR, "tmp.stm")
    stm_path = os.path.join(BASE_DIR, f"{evaluate_prefix}-{mode}.stm")
    out_path = os.path.join(BASE_DIR, f"out.{output_file}")

    _prepare_ctm(output_file, tmp_ctm, tmp2_ctm)
    _sort_stm(stm_path, tmp_stm)
    _run_command([sys.executable, os.path.join(BASE_DIR, "mergectmstm1.py"), tmp2_ctm, tmp_stm], cwd=BASE_DIR)
    shutil.copyfile(tmp2_ctm, out_path)

    try:
        return wer_calculation1(stm_path, out_path, isPrint)
    finally:
        if os.path.exists(out_path):
            os.remove(out_path)

def evaluate2(mode="dev", evaluate_prefix=None,
             output_file=None, isPrint=True):
    tmp_ctm = os.path.join(BASE_DIR, "tmp.ctm")
    tmp2_ctm = os.path.join(BASE_DIR, "tmp2.ctm")
    tmp_stm = os.path.join(BASE_DIR, "tmp.stm")
    stm_path = os.path.join(BASE_DIR, f"{evaluate_prefix}-{mode}.stm")
    out_path = os.path.join(BASE_DIR, f"out.{output_file}")

    _prepare_ctm(output_file, tmp_ctm, tmp2_ctm)
    _sort_stm(stm_path, tmp_stm)
    _run_command([sys.executable, os.path.join(BASE_DIR, "mergectmstm.py"), tmp2_ctm, tmp_stm], cwd=BASE_DIR)
    shutil.copyfile(tmp2_ctm, out_path)

    try:
        return wer_calculation(stm_path, out_path, isPrint)
    finally:
        if os.path.exists(out_path):
            os.remove(out_path)

def evaluate1(mode="dev", evaluate_prefix=None,
             output_file=None):
    return evaluate2(mode=mode, evaluate_prefix=evaluate_prefix, output_file=output_file, isPrint=True)

def evaluate4(mode="dev", evaluate_prefix=None,
             output_file=None):
    tmp_ctm = os.path.join(BASE_DIR, "tmp.ctm")
    tmp2_ctm = os.path.join(BASE_DIR, "tmp2.ctm")
    tmp_stm = os.path.join(BASE_DIR, "tmp.stm")
    stm_path = os.path.join(BASE_DIR, f"{evaluate_prefix}-{mode}.stm")
    out_path = os.path.join(BASE_DIR, f"out.{output_file}")

    _prepare_ctm(output_file, tmp_ctm, tmp2_ctm)
    _sort_stm(stm_path, tmp_stm)
    _run_command([sys.executable, os.path.join(BASE_DIR, "mergectmstm.py"), tmp2_ctm, tmp_stm], cwd=BASE_DIR)
    shutil.copyfile(tmp2_ctm, out_path)

    return wer_calculation2(stm_path, out_path, mode)

def evaluteMode(mode="dev", isPrint=True):
    if mode == 'dev':
        filePath = "./wer/dev/"
        path = 'wer.txt'

        fileReader = open(path, "w")

        fileList = os.listdir(filePath)
        fileList.sort(key=lambda x: int(x[21:25]))

        werResultList = []
        fileNameList = []
        for i, fileName in enumerate(fileList):
            path = os.path.join(filePath, fileName)
            shutil.copyfile(path, fileName)

            ret = evaluate1(
                        mode=mode, output_file=fileName,
                        evaluate_prefix='phoenix2014-groundtruth',
                )

            werResultList.append(ret)
            fileNameList.append(fileName)

            if os.path.exists(fileName):
                os.remove(fileName)

            fileReader.writelines(
                "{} {} {:.2f}\n".format(i,fileName,ret))

        indexValue = werResultList.index(min(werResultList))

        fileReader.writelines(
            "Min WER:index:{}, fileName:{}, WER:{:.2f}\n".format(indexValue, fileNameList[indexValue], werResultList[indexValue]))

        print("Min WER:index:{}, fileName:{}, WER:{:.2f}".format(indexValue, fileNameList[indexValue], werResultList[indexValue]))
    elif mode == 'test':
        filePath = "./wer/test/"
        path = 'wer.txt'

        fileReader = open(path, "w")

        fileList = os.listdir(filePath)
        fileList.sort(key=lambda x: int(x[22:26]))

        werResultList = []
        fileNameList = []
        for i, fileName in enumerate(fileList):
            path = os.path.join(filePath, fileName)
            shutil.copyfile(path, fileName)

            ret = evaluate1(
                mode=mode, output_file=fileName,
                evaluate_prefix='phoenix2014-groundtruth',
            )

            werResultList.append(ret)
            fileNameList.append(fileName)

            if os.path.exists(fileName):
                os.remove(fileName)

            fileReader.writelines(
                "{} {} {:.2f}\n".format(i, fileName, ret))

        indexValue = werResultList.index(min(werResultList))

        fileReader.writelines(
            "Min WER:index:{}, fileName:{}, WER:{:.2f}\n".format(indexValue, fileNameList[indexValue],
                                                                 werResultList[indexValue]))

        print("Min WER:index:{}, fileName:{}, WER:{:.2f}".format(indexValue, fileNameList[indexValue],
                                                                 werResultList[indexValue]))
    elif mode == 'evalute_dev':
        path = "evaluation/wer/evalute/output-hypothesis-dev.ctm"
        fileName = "output-hypothesis-dev.ctm"
        shutil.copyfile(path, os.path.join(BASE_DIR, fileName))

        mode = 'dev'
        ret = evaluate3(
            mode=mode, output_file=fileName,
            evaluate_prefix='phoenix2014-groundtruth',
            isPrint=isPrint,
        )

        temp_file = os.path.join(BASE_DIR, fileName)
        if os.path.exists(temp_file):
            os.remove(temp_file)
    elif mode == 'evalute_dev1':
        path = "evaluation/wer/evalute/output-hypothesis-dev.ctm"
        fileName = "output-hypothesis-dev.ctm"
        shutil.copyfile(path, os.path.join(BASE_DIR, fileName))

        mode = 'dev'
        ret = evaluate2(
            mode=mode, output_file=fileName,
            evaluate_prefix='phoenix2014-groundtruth',
            isPrint=isPrint,
        )

        temp_file = os.path.join(BASE_DIR, fileName)
        if os.path.exists(temp_file):
            os.remove(temp_file)
    elif mode == 'evalute_dev2':
        path = "evaluation/wer/evalute/output-hypothesis-dev1.ctm"
        fileName = "output-hypothesis-dev1.ctm"
        shutil.copyfile(path, os.path.join(BASE_DIR, fileName))

        mode = 'dev'
        ret = evaluate2(
            mode=mode, output_file=fileName,
            evaluate_prefix='phoenix2014-groundtruth',
            isPrint=isPrint,
        )

        temp_file = os.path.join(BASE_DIR, fileName)
        if os.path.exists(temp_file):
            os.remove(temp_file)
    elif mode == 'evalute_test':
        path = "evaluation/wer/evalute/output-hypothesis-test.ctm"
        fileName = "output-hypothesis-test.ctm"
        shutil.copyfile(path, os.path.join(BASE_DIR, fileName))

        mode = 'test'
        ret = evaluate2(
            mode=mode, output_file=fileName,
            evaluate_prefix='phoenix2014-groundtruth',
            isPrint=isPrint,
        )

        temp_file = os.path.join(BASE_DIR, fileName)
        if os.path.exists(temp_file):
            os.remove(temp_file)
    elif mode == 'evalute_train':
        path = "evaluation/wer/evalute/output-hypothesis-train.ctm"
        fileName = "output-hypothesis-train.ctm"
        shutil.copyfile(path, os.path.join(BASE_DIR, fileName))

        mode = 'train'
        ret = evaluate3(
            mode=mode, output_file=fileName,
            evaluate_prefix='phoenix2014-groundtruth',
            isPrint=isPrint,
        )

        temp_file = os.path.join(BASE_DIR, fileName)
        if os.path.exists(temp_file):
            os.remove(temp_file)

    return ret

if __name__ == '__main__':
    try:
        inputArgv = sys.argv[1]

        mode = inputArgv
    except:
        mode = 'test'

    evaluteMode(mode)
