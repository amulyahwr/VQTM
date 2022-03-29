import torch
import torch.nn as nn
from torch.autograd import Variable as Var
import torch.nn.init as init
import numpy as np
import torch.nn.functional as F

class VQTM(nn.Module):
    def __init__(self, args, vocab):
        super(VQTM, self).__init__()
        self.args = args
        self.in_dim = args.in_dim
        self.commitment_cost = args.commitment_cost
        self.vocab = vocab
        self.numbr_concepts = args.numbr_concepts

        #Embeddings
        self.emb = nn.Embedding(len(self.vocab), self.in_dim)
        self.emb.weight.requires_grad = False

        self.emb_concept = nn.Embedding(self.numbr_concepts, self.in_dim)
        self.emb_concept.weight.data.uniform_(-1/self.numbr_concepts, 1/self.numbr_concepts)

        self.heads = 2 #currently, supports 2, 4 and 8 heads
        self.W = nn.Parameter(torch.Tensor(self.heads, self.in_dim, self.in_dim).normal_(-0.05, 0.05))
        
        #LTS loss parameters
        self.criterion_lts = nn.HingeEmbeddingLoss(reduction='mean',margin=1)
        self.distance = nn.PairwiseDistance()
        self.concepts_idxs = []
        self.concepts_labels = []
        for i in range(self.args.numbr_concepts):
            for j in range(self.args.numbr_concepts):
                if i == j:
                    self.concepts_idxs.append(torch.unsqueeze(torch.LongTensor(np.array([i, j])),dim=0))
                    self.concepts_labels.append(1)
                else:
                    self.concepts_idxs.append(torch.unsqueeze(torch.LongTensor(np.array([i, j])),dim=0))
                    self.concepts_labels.append(-1)

        self.concepts_idxs = torch.cat(self.concepts_idxs, dim=0)
        self.concepts_labels = torch.FloatTensor(np.array(self.concepts_labels))

        if args.cuda:
            self.concepts_idxs = self.concepts_idxs.cuda()
            self.concepts_labels = self.concepts_labels.cuda()

        #output
        self.quant2vocab = nn.Linear(self.in_dim, len(self.vocab))
        self.ff_3 = nn.Linear(8 * self.in_dim, 4 * self.in_dim, bias=False)
        self.ff_2 = nn.Linear(4 * self.in_dim, 2 * self.in_dim, bias=False)
        self.ff_1 = nn.Linear(2 * self.in_dim, self.in_dim, bias=False)
    
    def calc_loss_lts(self):
        concept_embeds = self.emb_concept(self.concepts_idxs)
        distance = self.distance(concept_embeds[:,0,:], concept_embeds[:,1,:])
        loss = self.criterion_lts(distance, self.concepts_labels)
        return loss

    def multi(self, embedded_input):
        # multihead attention
        unnomralized_scores = torch.matmul(torch.matmul(embedded_input, self.W), self.emb_concept.weight.t())

        #Topic Proportion
        thetas = torch.softmax(unnomralized_scores, dim=2)


        theta = torch.mean(thetas, dim=0)
        theta = torch.sum(theta, dim=0, keepdim=True)
        theta = theta/torch.sum(theta)

        #Quantization
        quantized_words = torch.matmul(thetas, self.emb_concept.weight)

        quantized_words = torch.unbind(quantized_words, dim=0)
        quantized_words = torch.cat(quantized_words, dim=1)

        if self.heads == 2:
            quantized_words = torch.tanh(self.ff_1(quantized_words))
        elif self.heads == 4:
            quantized_words = torch.tanh(self.ff_2(quantized_words))
            quantized_words = torch.tanh(self.ff_1(quantized_words))
        elif self.heads == 8:
            quantized_words = torch.tanh(self.ff_3(quantized_words))
            quantized_words = torch.tanh(self.ff_2(quantized_words))
            quantized_words = torch.tanh(self.ff_1(quantized_words))


        # Loss
        e_latent_loss = torch.mean((quantized_words.detach() - embedded_input)**2)
        q_latent_loss = torch.mean((quantized_words - embedded_input.detach())**2)
        loss = q_latent_loss + self.commitment_cost * e_latent_loss
        quantized_words = embedded_input + (quantized_words - embedded_input).detach()

        quantized_docu = torch.mean(quantized_words, dim=0, keepdim=True)

        return thetas, quantized_words, quantized_docu, loss

    def forward(self, input_document):

        #embeddig of document
        embedded_document = self.emb(input_document)

        #Multi-VQTM
        thetas, quantized_words, quantized_docu, vq_loss = self.multi(embedded_document)
        
        #LTS loss
        lts_loss = self.calc_loss_lts()

        #quant2vocab
        res = self.quant2vocab(quantized_docu)
        res = torch.softmax(res, dim=1)
        outputs = torch.log(res+1e-6)

        #bin-count
        bin_count = torch.bincount(input_document, minlength=len(self.vocab)).float()

        outputs = outputs * bin_count

        return thetas, quantized_words, quantized_docu, outputs, vq_loss, lts_loss
