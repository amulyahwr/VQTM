from tqdm import tqdm
import math
import torch
from torch.autograd import Variable as Var
import torch.nn as nn

import numpy as np

class Trainer(object):
    def __init__(self, args, model, optimizer, vocab):
        super(Trainer, self).__init__()
        self.args       = args
        self.model      = model
        self.optimizer  = optimizer
        self.epoch      = 0
        self.vocab = vocab
        self.balance_factor = 0.001

    # helper function for training
    def train(self, input_documents):
        self.model.train()
        self.optimizer.zero_grad()
        total_tokens = 0
        batch_reconstr_loss = 0.0
        batch_vq_loss = 0.0
        batch_lts_loss = 0.0
        batch_loss = 0.0

        for docu in input_documents:

            if self.args.cuda:
                docu = docu.cuda()

            #outputs, theta, vq_loss = self.model(docu, "train")
            thetas, quantized_words, quantized_docu, outputs, vq_loss, lts_loss = self.model(docu, "train")
            
            #reconstruction loss
            total_tokens = total_tokens + len(docu)
            reconstr_loss = -torch.sum(outputs)
            batch_reconstr_loss = batch_reconstr_loss + reconstr_loss.item()

            #vq loss
            batch_vq_loss = batch_vq_loss + vq_loss.item()

            #lts loss
            lts_loss = self.balance_factor * lts_loss
            batch_lts_loss = batch_lts_loss + lts_loss.item()


            #Total Loss
            loss = reconstr_loss + vq_loss + lts_loss
            batch_loss = batch_loss + loss

        (batch_loss).backward()

        self.optimizer.step()
        self.optimizer.zero_grad()

        return batch_loss

    # helper function for testing
    def test(self, input_documents):
        self.model.eval()

        total_tokens = 0
        batch_reconstr_loss = 0.0
        batch_vq_loss = 0.0
        batch_lts_loss = 0.0
        batch_loss = 0.0

        with torch.no_grad():

            for docu in input_documents:

                if self.args.cuda:
                    docu = docu.cuda()

                #outputs, theta, vq_loss = self.model(docu, "test")
                thetas, quantized_words, quantized_docu, outputs, vq_loss, lts_loss = self.model(docu, "test")

                #reconstruction loss
                total_tokens = total_tokens + len(docu)
                reconstr_loss = -torch.sum(outputs)
                batch_reconstr_loss = batch_reconstr_loss + reconstr_loss.item()

                #vq_loss
                batch_vq_loss = batch_vq_loss + vq_loss.item()

                #lts loss
                lts_loss = self.balance_factor * lts_loss
                batch_lts_loss = batch_lts_loss + lts_loss.item()

                #Total Loss
                loss = reconstr_loss  + vq_loss + lts_loss
                batch_loss = batch_loss + loss

        return batch_reconstr_loss, batch_vq_loss, batch_lts_loss, batch_loss.item(), total_tokens
