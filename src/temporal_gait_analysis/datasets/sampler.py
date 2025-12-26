# datasets/sampler.py
import torch
from torch.utils.data import Sampler
import random
import copy
from collections import defaultdict

class TripletSampler(Sampler):
    def __init__(self, dataset, p_interval=8, k_samples=4):
        self.dataset = dataset
        self.p_interval = p_interval
        self.k_samples = k_samples
        self.batch_size = p_interval * k_samples
        
        # organiz index
        self.indices_by_label = defaultdict(list)
        for idx, label in enumerate(dataset.labels):
            self.indices_by_label[label].append(idx)
            
        self.labels = list(self.indices_by_label.keys())
        
        self.total_size = len(dataset)

        self.num_batches = self.total_size // self.batch_size
        self.total_samples = self.num_batches * self.batch_size

    def __len__(self):
        return self.total_samples

    def __iter__(self):

        indices_pool = copy.deepcopy(self.indices_by_label)
        
        for label in indices_pool:
            random.shuffle(indices_pool[label])
            
        final_indices = []

        while len(final_indices) < self.total_samples:
            
            random.shuffle(self.labels)
            
            # iterate over using P subjects
            for i in range(0, len(self.labels), self.p_interval):

                if len(final_indices) >= self.total_samples:
                    break
                
                # taking P subjects
                batch_labels = self.labels[i : i + self.p_interval]
                
                if len(batch_labels) < self.p_interval:
                    continue
                    
                batch_idxs = []
                for lbl in batch_labels:
                    # for every subj pick K videos
                    if len(indices_pool[lbl]) < self.k_samples:
                        indices_pool[lbl] = copy.deepcopy(self.indices_by_label[lbl])
                        random.shuffle(indices_pool[lbl])
                    

                    selected = indices_pool[lbl][:self.k_samples]

                    indices_pool[lbl] = indices_pool[lbl][self.k_samples:]
                    
                    batch_idxs.extend(selected)
                
                final_indices.extend(batch_idxs)
                
        return iter(final_indices)