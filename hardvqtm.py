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
    
    def calc_loss_lts(self):
        concept_embeds = self.emb_concept(self.concepts_idxs)
        distance = self.distance(concept_embeds[:,0,:], concept_embeds[:,1,:])
        loss = self.criterion_lts(distance, self.concepts_labels)
        return loss

    def hard(self, embedded_input):

        # Calculate distances
        distances = (torch.sum(embedded_input**2, dim=1, keepdim=True)
                    + torch.sum(self.emb_concept.weight**2, dim=1)
                    - 2 * torch.matmul(embedded_input, self.emb_concept.weight.t()))

        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self.numbr_concepts)
        if self.args.cuda:
            encodings = encodings.cuda()
        encodings.scatter_(1, encoding_indices, 1)

        # Quantize and unflatten
        quantized = torch.matmul(encodings, self.emb_concept.weight)

        # Loss
        e_latent_loss = torch.mean((quantized.detach() - embedded_input)**2)
        q_latent_loss = torch.mean((quantized - embedded_input.detach())**2)
        loss = q_latent_loss + self.commitment_cost * e_latent_loss

        quantized_words = embedded_input + (quantized - embedded_input).detach()
        quantized = torch.mean(quantized_words, dim=0, keepdim=True)


        return encodings, quantized_words, quantized, loss

    def forward(self, input_document):

        #embeddig of document
        embedded_document = self.emb(input_document)

        #HardVQTM
        thetas, quantized_words, quantized_docu, vq_loss = self.hard(embedded_document)
        
        #Loss lts
        lts_loss = self.calc_loss_lts()

        #quant2vocab
        res = self.quant2vocab(quantized_docu)
        res = torch.softmax(res, dim=1)
        outputs = torch.log(res+1e-6)

        #bin-count
        bin_count = torch.bincount(input_document, minlength=len(self.vocab)).float()

        outputs = outputs * bin_count

        return thetas, quantized_words, quantized_docu, outputs, vq_loss, lts_loss
