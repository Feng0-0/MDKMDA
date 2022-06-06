'''
@Author: your name
@Date: 2019-12-20 19:02:25
@LastEditTime: 2020-05-26 20:58:12
@LastEditors: Please set LastEditors
@Description: In User Settings Edit
@FilePath: /matengfei/KGCN_Keras-master/config.py
'''
# -*- coding: utf-8 -*-

import os

RAW_DATA_DIR = os.getcwd()+'/raw_data'
PROCESSED_DATA_DIR = os.getcwd()+'/data'
LOG_DIR = os.getcwd()+'/log'
MODEL_SAVED_DIR = os.getcwd()+'/ckpt'

KG_FILE = {'MDA':os.path.join(RAW_DATA_DIR,'MDA','train2id.txt')}
ENTITY2ID_FILE = {'MDA':os.path.join(RAW_DATA_DIR,'MDA','entity2id.txt')}
EXAMPLE_FILE = {'MDA':os.path.join(RAW_DATA_DIR,'MDA','approved_example.txt')}

# # add func sim
# MIRNA_FUNC_SIMILARITY_FILE = {'MDA':os.path.join(RAW_DATA_DIR,'MDA','miRNA_FUNC_similarity.txt')}
# DISEASE_SEMA_SIMILARITY_FILE = {'MDA':os.path.join(RAW_DATA_DIR,'MDA','disease_SEMA_similarity.txt')}

# #add gaussian similarity
# MIRNA_GAUS_SIMILARITY_FILE = {'MDA':os.path.join(RAW_DATA_DIR,'MDA','miRNA_GAUS_similarity.txt')}
# DISEASE_GAUS_SIMILARITY_FILE = {'MDA':os.path.join(RAW_DATA_DIR,'MDA','disease_GAUS_similarity.txt')}

# add fusion similarity
MIRNA_FUNC_GAUS_SIMILARITY_FILE = {'MDA':os.path.join(RAW_DATA_DIR,'MDA','mir_func_sim.txt')}
DISEASE_SEMA_GAUS_SIMILARITY_FILE = {'MDA':os.path.join(RAW_DATA_DIR,'MDA','disease_sema_sim.txt')}

#add func and semantic similarity
MIRNA_SEMA_SIMILARITY_FILE = {'MDA':os.path.join(RAW_DATA_DIR,'MDA','mir_sema_sim.txt')}
DISEASE_FUNC_SIMILARITY_FILE = {'MDA':os.path.join(RAW_DATA_DIR,'MDA','disease_func_sim.txt')}

SEPARATOR = {'MDA':'\t'}
THRESHOLD = {'MDA':4} #添加drug�?�?
NEIGHBOR_SIZE = {'MDA':16}

#
# DRUG_VOCAB_TEMPLATE = '{dataset}_drug_vocab.pkl'
ENTITY_VOCAB_TEMPLATE = '{dataset}_entity_vocab.pkl'
RELATION_VOCAB_TEMPLATE = '{dataset}_relation_vocab.pkl'
ADJ_ENTITY_TEMPLATE = '{dataset}_adj_entity.npy'
ADJ_RELATION_TEMPLATE = '{dataset}_adj_relation.npy'
TRAIN_DATA_TEMPLATE = '{dataset}_train.npy'
DEV_DATA_TEMPLATE = '{dataset}_dev.npy'
TEST_DATA_TEMPLATE = '{dataset}_test.npy'
#RESULT_LOG='result.txt'
RESULT_LOG={'MDA':'MDA_result.txt','kegg':'kegg_result.txt'}
PERFORMANCE_LOG = 'kgcn_performance.log'
DRUG_EXAMPLE='{dataset}_examples.npy'

class ModelConfig(object):
    def __init__(self):
        self.neighbor_sample_size = 4 # neighbor sampling size
        self.embed_dim = 32  # dimension of embedding
        self.n_depth = 2    # depth of receptive field
        self.l2_weight = 1e-7  # l2 regularizer weight
        self.lr = 2e-2  # learning rate
        self.batch_size = 65536
        self.aggregator_type = 'sum'
        self.n_epoch = 50
        self.optimizer = 'adam'

        self.drug_vocab_size = None
        self.entity_vocab_size = None
        self.relation_vocab_size = None
        self.adj_entity = None
        self.adj_relation = None

        self.exp_name = None
        self.model_name = None

        # checkpoint configuration 设置检查点
        self.checkpoint_dir = MODEL_SAVED_DIR
        self.checkpoint_monitor = 'val_auc'
        self.checkpoint_save_best_only = True
        self.checkpoint_save_weights_only = True
        self.checkpoint_save_weights_mode = 'max'
        self.checkpoint_verbose = 1

        # early_stoping configuration
        self.early_stopping_monitor = 'val_auc'
        self.early_stopping_mode = 'max'
        self.early_stopping_patience = 5
        self.early_stopping_verbose = 1
        self.dataset='drug'
        self.K_Fold=1
        self.callbacks_to_add = None

        # config for learning rating scheduler and ensembler
        self.swa_start = 3
