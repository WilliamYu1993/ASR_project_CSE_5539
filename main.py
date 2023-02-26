import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchcontrib.optim import SWA
from torch.nn.utils import rnn
from torch.utils.tensorboard import SummaryWriter
from conformer.model import Conformer
from model import *

import os
import sys
import pdb
import json
import argparse
import torchaudio
from argparse import Namespace
import numpy as np
from tqdm import tqdm
from colorama import Fore
from collections import OrderedDict
from multiprocessing import Pool
from torch.utils import data

train_dataset = torchaudio.datasets.LIBRISPEECH("./", url="train-clean-100", download=True)
test_dataset = torchaudio.datasets.LIBRISPEECH("./", url="test-clean", download=True)
#torchaudio.transforms.FrequencyMasking()
#torchaudio.transforms.TimeMasking()

def avg_wer(wer_scores, combined_ref_len):
    return float(sum(wer_scores)) / float(combined_ref_len)


def _levenshtein_distance(ref, hyp):
    """Levenshtein distance is a string metric for measuring the difference
    between two sequences. Informally, the levenshtein disctance is defined as
    the minimum number of single-character edits (substitutions, insertions or
    deletions) required to change one word into the other. We can naturally
    extend the edits to word level when calculate levenshtein disctance for
    two sentences.
    """
    m = len(ref)
    n = len(hyp)

    # special case
    if ref == hyp:
        return 0
    if m == 0:
        return n
    if n == 0:
        return m

    if m < n:
        ref, hyp = hyp, ref
        m, n = n, m

    # use O(min(m, n)) space
    distance = np.zeros((2, n + 1), dtype=np.int32)

    # initialize distance matrix
    for j in range(0,n + 1):
        distance[0][j] = j

    # calculate levenshtein distance
    for i in range(1, m + 1):
        prev_row_idx = (i - 1) % 2
        cur_row_idx = i % 2
        distance[cur_row_idx][0] = i
        for j in range(1, n + 1):
            if ref[i - 1] == hyp[j - 1]:
                distance[cur_row_idx][j] = distance[prev_row_idx][j - 1]
            else:
                s_num = distance[prev_row_idx][j - 1] + 1
                i_num = distance[cur_row_idx][j - 1] + 1
                d_num = distance[prev_row_idx][j] + 1
                distance[cur_row_idx][j] = min(s_num, i_num, d_num)

    return distance[m % 2][n]


def word_errors(reference, hypothesis, ignore_case=False, delimiter=' '):
    """Compute the levenshtein distance between reference sequence and
    hypothesis sequence in word-level.
    :param reference: The reference sentence.
    :type reference: basestring
    :param hypothesis: The hypothesis sentence.
    :type hypothesis: basestring
    :param ignore_case: Whether case-sensitive or not.
    :type ignore_case: bool
    :param delimiter: Delimiter of input sentences.
    :type delimiter: char
    :return: Levenshtein distance and word number of reference sentence.
    :rtype: list
    """
    if ignore_case == True:
        reference = reference.lower()
        hypothesis = hypothesis.lower()

    ref_words = reference.split(delimiter)
    hyp_words = hypothesis.split(delimiter)

    edit_distance = _levenshtein_distance(ref_words, hyp_words)
    return float(edit_distance), len(ref_words)


def char_errors(reference, hypothesis, ignore_case=False, remove_space=False):
    """Compute the levenshtein distance between reference sequence and
    hypothesis sequence in char-level.
    :param reference: The reference sentence.
    :type reference: basestring
    :param hypothesis: The hypothesis sentence.
    :type hypothesis: basestring
    :param ignore_case: Whether case-sensitive or not.
    :type ignore_case: bool
    :param remove_space: Whether remove internal space characters
    :type remove_space: bool
    :return: Levenshtein distance and length of reference sentence.
    :rtype: list
    """
    if ignore_case == True:
        reference = reference.lower()
        hypothesis = hypothesis.lower()

    join_char = ' '
    if remove_space == True:
        join_char = ''

    reference = join_char.join(filter(None, reference.split(' ')))
    hypothesis = join_char.join(filter(None, hypothesis.split(' ')))

    edit_distance = _levenshtein_distance(reference, hypothesis)
    return float(edit_distance), len(reference)


def wer(reference, hypothesis, ignore_case=False, delimiter=' '):
    """Calculate word error rate (WER). WER compares reference text and
    hypothesis text in word-level. WER is defined as:
    .. math::
        WER = (Sw + Dw + Iw) / Nw
    where
    .. code-block:: text
        Sw is the number of words subsituted,
        Dw is the number of words deleted,
        Iw is the number of words inserted,
        Nw is the number of words in the reference
    We can use levenshtein distance to calculate WER. Please draw an attention
    that empty items will be removed when splitting sentences by delimiter.
    :param reference: The reference sentence.
    :type reference: basestring
    :param hypothesis: The hypothesis sentence.
    :type hypothesis: basestring
    :param ignore_case: Whether case-sensitive or not.
    :type ignore_case: bool
    :param delimiter: Delimiter of input sentences.
    :type delimiter: char
    :return: Word error rate.
    :rtype: float
    :raises ValueError: If word number of reference is zero.
    """
    edit_distance, ref_len = word_errors(reference, hypothesis, ignore_case,
                                         delimiter)

    if ref_len == 0:
        raise ValueError("Reference's word number should be greater than 0.")

    wer = float(edit_distance) / ref_len
    return wer


def cer(reference, hypothesis, ignore_case=False, remove_space=False):
    """Calculate charactor error rate (CER). CER compares reference text and
    hypothesis text in char-level. CER is defined as:
    .. math::
        CER = (Sc + Dc + Ic) / Nc
    where
    .. code-block:: text
        Sc is the number of characters substituted,
        Dc is the number of characters deleted,
        Ic is the number of characters inserted
        Nc is the number of characters in the reference
    We can use levenshtein distance to calculate CER. Chinese input should be
    encoded to unicode. Please draw an attention that the leading and tailing
    space characters will be truncated and multiple consecutive space
    characters in a sentence will be replaced by one space character.
    :param reference: The reference sentence.
    :type reference: basestring
    :param hypothesis: The hypothesis sentence.
    :type hypothesis: basestring
    :param ignore_case: Whether case-sensitive or not.
    :type ignore_case: bool
    :param remove_space: Whether remove internal space characters
    :type remove_space: bool
    :return: Character error rate.
    :rtype: float
    :raises ValueError: If the reference length is zero.
    """
    edit_distance, ref_len = char_errors(reference, hypothesis, ignore_case,
                                         remove_space)

    if ref_len == 0:
        raise ValueError("Length of reference should be greater than 0.")

    cer = float(edit_distance) / ref_len
    return cer

class TextTransform:
    """Maps characters to integers and vice versa"""
    def __init__(self):
        char_map_str = """
        ' 0
        <SPACE> 1
        a 2
        b 3
        c 4
        d 5
        e 6
        f 7
        g 8
        h 9
        i 10
        j 11
        k 12
        l 13
        m 14
        n 15
        o 16
        p 17
        q 18
        r 19
        s 20
        t 21
        u 22
        v 23
        w 24
        x 25
        y 26
        z 27
        """
        self.char_map = {}
        self.index_map = {}
        for line in char_map_str.strip().split('\n'):
            ch, index = line.split()
            self.char_map[ch] = int(index)
            self.index_map[int(index)] = ch
        self.index_map[1] = ' '

    def text_to_int(self, text):
        """ Use a character map and convert text to an integer sequence """
        int_sequence = []
        for c in text:
            if c == ' ':
                ch = self.char_map['<SPACE>']
            else:
                ch = self.char_map[c]
            int_sequence.append(ch)
        return int_sequence

    def int_to_text(self, labels):
        """ Use a character map and convert integer labels to an text sequence """
        string = []
        for i in labels:
            string.append(self.index_map[i])
        return ''.join(string).replace('<SPACE>', ' ')

train_audio_transforms = nn.Sequential(
    torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_mels=128),
    torchaudio.transforms.FrequencyMasking(freq_mask_param=15),
    torchaudio.transforms.TimeMasking(time_mask_param=35)
)

valid_audio_transforms = torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_mels=128)

text_transform = TextTransform()

def data_processing(data, data_type="train"):
    spectrograms = []
    labels = []
    t_input_lengths = []
    input_lengths = []
    label_lengths = []
    for (waveform, _, utterance, _, _, _) in data:
        if data_type == 'train':
            spec = train_audio_transforms(waveform).squeeze(0).transpose(0, 1)
        else:
            spec = valid_audio_transforms(waveform).squeeze(0).transpose(0, 1)
        spectrograms.append(spec)
        label = torch.Tensor(text_transform.text_to_int(utterance.lower()))
        labels.append(label)
        t_input_lengths.append(spec.shape[0])
        input_lengths.append(spec.shape[0]//2)
        label_lengths.append(len(label))

    spectrograms = nn.utils.rnn.pad_sequence(spectrograms, batch_first=True).unsqueeze(1).transpose(2, 3)
    labels = nn.utils.rnn.pad_sequence(labels, batch_first=True)

    return spectrograms, labels, t_input_lengths, input_lengths, label_lengths

def GreedyDecoder(output, labels, label_lengths, blank_label=28, collapse_repeated=True):
    arg_maxes = torch.argmax(output, dim=2)
    decodes = []
    targets = []
    for i, args in enumerate(arg_maxes):
        decode = []
        targets.append(text_transform.int_to_text(labels[i][:label_lengths[i]].tolist()))
        for j, index in enumerate(args):
            if index != blank_label:
                if collapse_repeated and j != 0 and index == args[j -1]:
                    continue
                decode.append(index.item())
        decodes.append(text_transform.int_to_text(decode))
    return decodes, targets

class IterMeter(object):
    """keeps track of total iterations"""
    def __init__(self):
        self.val = 0

    def step(self):
        self.val += 1

    def get(self):
        return self.val



def rnn_collate(batch):

    n = rnn.pad_sequence([b[0] for b in batch]).transpose(0, 1)
    c = rnn.pad_sequence([b[1] for b in batch]).transpose(0, 1)
    l = torch.LongTensor([b[2] for b in batch])

    return n, c, l

def fix_seed(seed):
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def cal_pesq(x, y, l):
    try:
        score = pesq(16000, y, x, 'wb')
    except:
        score = 0.
    #print(score)
    del x, y
    return score

def evaluate(x, y, lens, fn):
    y = list([y])#(y.cpu().detach().numpy())
    x = list([x])#(x.cpu().detach().numpy())
    lens = list(lens)
    pool = Pool(processes=args.num_workers)
    try:
        ret = pool.starmap(
            fn,
            iter([(deg, ref, l) for deg, ref, l in zip(x, y, lens)])
        )
        pool.close()
        del x, y
        return torch.FloatTensor(ret).mean()

    except KeyboardInterrupt:
        pool.terminate()
        pool.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # system setting
    parser.add_argument('--exp_dir', default=os.getcwd(), type=str)
    parser.add_argument('--exp_name', default='logs', type=str)
    parser.add_argument('--data_dir', default='./', type=str)
    parser.add_argument('--num_workers', default=3, type=int)
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--add_graph', action='store_true')
    parser.add_argument('--log_interval', default=20, type=int)
    parser.add_argument('--seed', default=999, type=int)

    # training specifics
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--learning_rate', default=1e-3, type=float)
    parser.add_argument('--num_epochs', default=100, type=int)
    parser.add_argument('--grad_accumulate_batches', default=1, type=int)
    parser.add_argument('--log_grad_norm', action='store_true')
    parser.add_argument('--resume_dir', default='', type=str)
    parser.add_argument('--use_swa', action='store_true')
    parser.add_argument('--use_logstftmagloss', action='store_true')
    parser.add_argument('--lr_decay', default=1.0, type=float)

    # stft/istft settings
    parser.add_argument('--n_fft', default=512, type=int)
    parser.add_argument('--hop_length', default=256, type=int)

    args = parser.parse_args()

    # add hyperparameters
    ckpt_path = os.path.join(args.exp_dir, args.exp_name, 'ckpt')
    if not os.path.exists(ckpt_path):
        os.makedirs(ckpt_path)
        os.makedirs(ckpt_path.replace('ckpt', 'logs'))
        with open(os.path.join(ckpt_path, 'hparams.json'), 'w') as f:
            json.dump(vars(args), f)
    else:
        print(f'Experiment {args.exp_name} already exists.')
        sys.exit()

    writer = SummaryWriter(os.path.join(args.exp_dir, args.exp_name, 'logs'))
    writer.add_hparams(vars(args), dict())

    # seed
    if args.seed:
        fix_seed(args.seed)

    # device
    use_cuda = torch.cuda.is_available()
    torch.manual_seed(7)
    device = 'cuda' if torch.cuda.is_available() and args.cuda else 'cpu'

    if device == 'cuda':
        print(f'DEVICE: [{torch.cuda.current_device()}] {torch.cuda.get_device_name()}')
    else:
        print(f'DEVICE: CPU')

    if not os.path.isdir("./data"):
        os.makedirs("./data")

    hparams = {
        "n_cnn_layers": 3,
        "n_rnn_layers": 5,
        "rnn_dim": 512,
        "n_class": 29,
        "n_feats": 128,
        "stride": 2,
        "dropout": 0.1,
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        "epochs": args.num_epochs
    }

    train_dataset = torchaudio.datasets.LIBRISPEECH("./data", url="train-clean-100", download=True)
    test_dataset = torchaudio.datasets.LIBRISPEECH("./data", url="test-clean", download=True)

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    train_dataloader = data.DataLoader(dataset=train_dataset,
                                batch_size=hparams['batch_size'],
                                shuffle=True,
                                collate_fn=lambda x: data_processing(x, 'train'),
                                **kwargs)
    test_dataloader = data.DataLoader(dataset=test_dataset,
                                batch_size=hparams['batch_size'],
                                shuffle=False,
                                collate_fn=lambda x: data_processing(x, 'valid'),
                                **kwargs)
    """
    asr_net = SpeechRecognitionModel(
        hparams['n_cnn_layers'], hparams['n_rnn_layers'], hparams['rnn_dim'],
        hparams['n_class'], hparams['n_feats'], hparams['stride'], hparams['dropout']
        ).to(device)
    """
    asr_net = Conformer(num_classes=hparams['n_class'],
                  input_dim=hparams['n_feats'],
                  encoder_dim=32,
                  num_encoder_layers=10).to(device)
     
    optimizer = optim.Adam(asr_net.parameters(), lr=hparams["learning_rate"], weight_decay=args.lr_decay)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=hparams['learning_rate'],
                                            steps_per_epoch=int(len(train_dataloader)),
                                            epochs=hparams['epochs'],
                                            anneal_strategy='linear')
    iter_meter = IterMeter()
    # add graph to tensorboard
    if args.add_graph:
        dummy = torch.randn(16, 1, args.hop_length * 16).to(device)
        writer.add_graph(asr_net, dummy)

    criterion = nn.CTCLoss().to(device)
    start_epoch = 0
    total_loss = 0.0
    best_wer = 1.01
    data_len = len(train_dataloader.dataset)
    for epoch in range(start_epoch, start_epoch + args.num_epochs, 1):
        # ------------- training -------------
        asr_net.train()
        pbar = tqdm(train_dataloader, bar_format='{l_bar}%s{bar}%s{r_bar}'%(Fore.BLUE, Fore.RESET), ascii=True)
        pbar.set_description(f'Epoch {epoch + 1}')
        total_loss = 0.0
        if args.log_grad_norm:
            total_norm = 0.0
        asr_net.zero_grad()
        i=0
        for batch_idx, _data in enumerate(pbar):
            spectrograms, labels, t_input_lengths, input_lengths, label_lengths = _data
            spectrograms, labels = spectrograms.to(device), labels.to(device)
            optimizer.zero_grad()
            output, out_lengths = asr_net(spectrograms.squeeze(1).squeeze(1).permute(0,2,1), torch.LongTensor(t_input_lengths))  # (batch, time, n_class)
            #ou=asr_net(spectrograms)
            #pdb.set_trace()
            output = output.permute(1,0,2) # (time, batch, n_class)
            #pdb.set_trace()
            loss = criterion(output, labels, out_lengths, torch.LongTensor(label_lengths))
            loss.backward()
            if math.isinf(loss):
                print(out_lengths, output.shape, label_lengths)
            if output.shape[0]>(labels.shape[-1]+10):
                optimizer.step()
            #else:
                #print(label_lengths, out_lengths)
            scheduler.step()
            iter_meter.step()
            # if batch_idx % 100 == 0 or batch_idx == data_len:
            #     print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            #         epoch, batch_idx * len(spectrograms), data_len,
            #         100. * batch_idx / len(train_loader), loss.item()))


            # log metrics
            pbar_dict = OrderedDict({
                'loss': loss.item(),
            })
            pbar.set_postfix(pbar_dict)

            total_loss += loss.item()
            #pdb.set_trace()
            if (i + 1) % args.log_interval == 0:
                step = epoch * len(train_dataloader) + i
                writer.add_scalar('Loss/train', total_loss / args.log_interval, step+1)
                total_loss = 0.0

                # log gradient norm
                if args.log_grad_norm:
                    for p in asr_net.parameters():
                        if p.requires_grad:
                            try:
                                norm = p.grad.data.norm(2)
                                total_norm += norm.item() ** 2
                            except:
                                pass

                    norm = total_norm ** 0.5
                    writer.add_scalar('Gradient 2-Norm/train', norm, step+1)
                    total_norm = 0.0
            i=i+1
        # ------------- validation -------------
        pbar = tqdm(test_dataloader, bar_format='{l_bar}%s{bar}%s{r_bar}'%(Fore.LIGHTMAGENTA_EX, Fore.RESET), ascii=True)
        pbar.set_description('Validation')
        test_loss, total_pesq = 0.0, 0.0
        test_cer=[]
        test_wer=[]
        num_test_data = len(test_dataloader)
        with torch.no_grad():
            asr_net.eval()
            for i, (_data) in enumerate(pbar):

                spectrograms, labels, t_input_lengths, input_lengths, label_lengths = _data
                spectrograms, labels = spectrograms.to(device), labels.to(device)
                #pdb.set_trace()
                output, out_lengths = asr_net(spectrograms.squeeze(1).squeeze(1).permute(0,2,1), torch.LongTensor(t_input_lengths))  # (batch, time, n_class)
                output = F.log_softmax(output, dim=2)
                output = output.permute(1,0,2) # (time, batch, n_class)

                loss = criterion(output, labels, out_lengths, torch.LongTensor(label_lengths))
                if math.isinf(loss.item()):
                    test_loss += 10.0 / len(test_dataloader)
                else:
                    test_loss += loss.item() / len(test_dataloader)

                decoded_preds, decoded_targets = GreedyDecoder(output.transpose(0, 1), labels, label_lengths)
                for j in range(len(decoded_preds)):
                    c_score = cer(decoded_targets[j], decoded_preds[j])
                    w_score = wer(decoded_targets[j], decoded_preds[j])
                    test_cer.append(c_score)
                    test_wer.append(w_score)
                    pbar_dict = OrderedDict({
                    'test_loss': loss.item(),
                    'val_cer': c_score,
                    'val_wer': w_score,
                    })
                    pbar.set_postfix(pbar_dict)
                
            if scheduler is not None:
                scheduler.step(test_loss)

            avg_cer = sum(test_cer)/len(test_cer)
            avg_wer = sum(test_wer)/len(test_wer)
            
            print('Test set: Average loss: {:.4f}, Average CER: {:4f} Average WER: {:.4f}\n'.format(test_loss, avg_cer, avg_wer))

            writer.add_scalar('Loss/valid', total_loss, epoch)

            # checkpointing
            curr_wer = avg_wer
            if  curr_wer < best_wer:# or ((epoch+1)%5)==0:
                best_wer = curr_wer
                save_path = os.path.join(ckpt_path, 'model_best.ckpt')
                print(f'Saving checkpoint to {save_path}')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': asr_net.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': test_loss
                }, save_path)

    writer.flush()
    writer.close()

