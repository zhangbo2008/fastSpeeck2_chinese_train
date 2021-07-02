













# ==================第一步: 文件写入raw_path

import argparse
# https://www.openslr.org/resources/93/data_aishell3.tgz  扔迅雷里面很快.
import yaml

from preprocessor import ljspeech, aishell3, libritts


def main(config):
    if "LJSpeech" in config["dataset"]:
        ljspeech.prepare_align(config)
    if "AISHELL3" in config["dataset"]:
        aishell3.prepare_align(config)
    if "LibriTTS" in config["dataset"]:
        libritts.prepare_align(config)

if 0:
    if __name__ == "__main__":
        parser = argparse.ArgumentParser()

        args = parser.parse_args()
        args.config='config/AISHELL3/preprocess.yaml'


        config = yaml.load(open(args.config, "r"), Loader=yaml.FullLoader)
        config['path']['lexicon_path'] = 'lexicon/madarin_lexicon.txt'
        config['path']['corpus_path']='AISHELL-3-Sample'
        config['path']['raw_path']='raw_path/AISHELL-3-Sample'
        # config['path']['lexicon_path']='AISHELL-3-Sample/train/label_train-set.txt'
        config['dataset']='AISHELL3'
        main(config)

#=====================现在文件都写入了raw_path里面.




#============开启第二步/  mfa111
#  https://www.bilibili.com/read/cv7351673/
# ./montreal-forced-aligner/bin/mfa_align raw_data/LJSpeech/ lexicon/librispeech-lexicon.txt english preprocessed_data/LJSpeech

# cmd 运行
# ./montreal-forced-aligner/bin/mfa_align raw_path/AISHELL-3-Sample/  lexicon/pinyin-lexicon-r.txt english preprocessed_data/LJSpeech

# https://montreal-forced-aligner.readthedocs.io/en/latest/pretrained_models.html
#========如果有chinese model就用这个==========我们可以下载到.


#=====================目前使用这个...........
#  E:\Users\Administrator\PycharmProjects\fairseq-gec\FastSpeech2\montreal-forced-aligner\bin\mfa_align.exe  raw_path/AISHELL-3-Sample/  lexicon/madarin_lexicon.txt mandarin.zip   preprocessed_data/atshell/TextGrid

#=====================其他东西.
# 字典下载:https://github.com/Jackiexiao/MTTS/blob/master/misc/mandarin-for-montreal-forced-aligner-pre-trained-model.lexicon

#=====有bug 自己训练一个.   E:\Users\Administrator\PycharmProjects\fairseq-gec\FastSpeech2\montreal-forced-aligner\bin\mfa_train_and_align.exe raw_path/AISHELL-3-Sample/  lexicon/madarin_lexicon.txt   preprocessed_data/atshell



#=====有bug 自己训练一个.   E:\Users\Administrator\PycharmProjects\fairseq-gec\FastSpeech2\montreal-forced-aligner\bin\mfa_align.exe raw_path/AISHELL-3-Sample/SSB0009  lexicon/madarin_lexicon.txt   mandarin_pinyin_g2p.zip   test1




#==========第三部python3 preprocess.py config/LJSpeech/preprocess.yaml




import argparse

import yaml

from preprocessor.preprocessor import Preprocessor

if 1:
    if __name__ == "__main__":
        parser = argparse.ArgumentParser()
        # parser.add_argument("config", type=str, help="path to preprocess.yaml")
        args = parser.parse_args()

        args.config='config/AISHELL3/preprocess.yaml'
        config = yaml.load(open(args.config, "r"), Loader=yaml.FullLoader)
        config['path']['corpus_path']='AISHELL-3-Sample'
        config['path']['raw_path']='raw_path/AISHELL-3-Sample'
        config['path']['preprocessed_path']='preprocessed_data/atshell'
        config['path']['lexicon_path']='lexicon/madarin_lexicon.txt'
        config["preprocessing"]["val_size"]=0 # 不要测试数据.
        # config['path']['lexicon_path']='AISHELL-3-Sample/train/label_train-set.txt'
        config['dataset']='AISHELL3'


        preprocessor = Preprocessor(config)
        preprocessor.build_from_path()










#=========================最后一步......训练!!!!!!!!!!!!!!!!!!
import argparse
import os

import torch
import yaml
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from utils.model import get_model, get_vocoder, get_param_num
from utils.tools import to_device, log, synth_one_sample
from model import FastSpeech2Loss
from dataset import Dataset

from evaluate import evaluate

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main(args, configs):
    print("Prepare training ...")

    preprocess_config, model_config, train_config = configs

    # Get dataset
    dataset = Dataset(
        "train.txt", preprocess_config, train_config, sort=True, drop_last=True
    )
    batch_size = train_config["optimizer"]["batch_size"]
    group_size = 1  # Set this larger than 1 to enable sorting in Dataset
    assert batch_size * group_size < len(dataset)
    loader = DataLoader(
        dataset,
        batch_size=batch_size * group_size,
        shuffle=True,
        collate_fn=dataset.collate_fn,
    )

    # Prepare model
    model, optimizer = get_model(args, configs, device, train=True)
    model = nn.DataParallel(model)
    num_param = get_param_num(model)
    Loss = FastSpeech2Loss(preprocess_config, model_config).to(device)
    print("Number of FastSpeech2 Parameters:", num_param)

    # Load vocoder
    vocoder = get_vocoder(model_config, device)

    # Init logger
    for p in train_config["path"].values():
        os.makedirs(p, exist_ok=True)
    train_log_path = os.path.join(train_config["path"]["log_path"], "train")
    val_log_path = os.path.join(train_config["path"]["log_path"], "val")
    os.makedirs(train_log_path, exist_ok=True)
    os.makedirs(val_log_path, exist_ok=True)
    train_logger = SummaryWriter(train_log_path)
    val_logger = SummaryWriter(val_log_path)

    # Training
    step = args.restore_step + 1
    epoch = 1
    grad_acc_step = train_config["optimizer"]["grad_acc_step"]
    grad_clip_thresh = train_config["optimizer"]["grad_clip_thresh"]
    total_step = train_config["step"]["total_step"]
    log_step = train_config["step"]["log_step"]
    save_step = train_config["step"]["save_step"]
    synth_step = train_config["step"]["synth_step"]
    val_step = train_config["step"]["val_step"]

    outer_bar = tqdm(total=total_step, desc="Training", position=0)
    outer_bar.n = args.restore_step
    outer_bar.update()

    while True:
        inner_bar = tqdm(total=len(loader), desc="Epoch {}".format(epoch), position=1)
        for batchs in loader:
            for batch in batchs:
                batch = to_device(batch, device)
                '''
                sample = {
              ids,   声音文件名
            raw_texts,  # 是拼音
            speakers,
            texts,
            text_lens,
            max(text_lens),
            mels,
            mel_lens,
            max(mel_lens),
            pitches,
            energies,
            durations,
        }
                '''
                # Forward
                output = model(*(batch[2:]))

                # Cal Loss
                losses = Loss(batch, output)
                total_loss = losses[0]

                # Backward
                total_loss = total_loss / grad_acc_step
                total_loss.backward()
                if step % grad_acc_step == 0:
                    # Clipping gradients to avoid gradient explosion
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip_thresh)
# optimizer.step()，模型才会更新，而scheduler.step()是对lr进行调整   因为如果不清零，那么使用的这个grad就得同上一个mini-batch有关，这不是我们需要的结果。再回过头来看，我们知道optimizer更新参数空间需要基于反向梯度，因此，当调用optimizer.step()的时候应当是loss.backward()的时候，
                    # Update weights
                    optimizer.step_and_update_lr()
                    optimizer.zero_grad()

                if step % log_step == 0:
                    losses = [l.item() for l in losses]
                    message1 = "Step {}/{}, ".format(step, total_step)
                    message2 = "Total Loss: {:.4f}, Mel Loss: {:.4f}, Mel PostNet Loss: {:.4f}, Pitch Loss: {:.4f}, Energy Loss: {:.4f}, Duration Loss: {:.4f}".format(
                        *losses
                    )

                    with open(os.path.join(train_log_path, "log.txt"), "a") as f:
                        f.write(message1 + message2 + "\n")

                    outer_bar.write(message1 + message2)

                    log(train_logger, step, losses=losses)

                if step % synth_step == 0:
                    fig, wav_reconstruction, wav_prediction, tag = synth_one_sample(
                        batch,
                        output,
                        vocoder,
                        model_config,
                        preprocess_config,
                    )
                    log(
                        train_logger,
                        fig=fig,
                        tag="Training/step_{}_{}".format(step, tag),
                    )
                    sampling_rate = preprocess_config["preprocessing"]["audio"][
                        "sampling_rate"
                    ]
                    log(
                        train_logger,
                        audio=wav_reconstruction,
                        sampling_rate=sampling_rate,
                        tag="Training/step_{}_{}_reconstructed".format(step, tag),
                    )
                    log(
                        train_logger,
                        audio=wav_prediction,
                        sampling_rate=sampling_rate,
                        tag="Training/step_{}_{}_synthesized".format(step, tag),
                    )

                if step % val_step == 0:
                    model.eval()
                    message = evaluate(model, step, configs, val_logger, vocoder)
                    with open(os.path.join(val_log_path, "log.txt"), "a") as f:
                        f.write(message + "\n")
                    outer_bar.write(message)

                    model.train()

                if step % save_step == 0:
                    torch.save(
                        {
                            "model": model.module.state_dict(),
                            "optimizer": optimizer._optimizer.state_dict(),
                        },
                        os.path.join(
                            train_config["path"]["ckpt_path"],
                            "{}.pth.tar".format(step),
                        ),
                    )

                if step == total_step:
                    quit()
                step += 1
                outer_bar.update(1)

            inner_bar.update(1)
        epoch += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--restore_step", type=int, default=0)
    parser.add_argument(
        "-p",
        "--preprocess_config",
        type=str,
        required=False,
        help="path to preprocess.yaml",
    )
    parser.add_argument(
        "-m", "--model_config", type=str, required=False, help="path to model.yaml"
    )
    parser.add_argument(
        "-t", "--train_config", type=str, required=False, help="path to train.yaml"
    )
    args = parser.parse_args()

    args.preprocess_config='config/AISHELL3/preprocess.yaml'
    config = yaml.load(open(args.preprocess_config, "r"), Loader=yaml.FullLoader)
    config['path']['corpus_path']='AISHELL-3-Sample'
    config['path']['raw_path']='raw_path/AISHELL-3-Sample'
    config['path']['preprocessed_path']='preprocessed_data/atshell'
    # config['path']['lexicon_path']='AISHELL-3-Sample/train/label_train-set.txt'
    config['dataset']='AISHELL3'
    config['path']['lexicon_path'] = 'lexicon/madarin_lexicon.txt'
    args.m='config/AISHELL3/model.yaml '
    args.t='config/AISHELL3/train.yaml '
    preprocess_config=config







    # Read Config
    # preprocess_config = yaml.load(
    #     open(args.preprocess_config, "r"), Loader=yaml.FullLoader
    # )
    model_config = yaml.load(open(args.m, "r"), Loader=yaml.FullLoader)
    train_config = yaml.load(open(args.t, "r"), Loader=yaml.FullLoader)
    train_config['optimizer']['batch_size']=2 # 数据集小我就开小.
    configs = (preprocess_config, model_config, train_config)

    main(args, configs)

