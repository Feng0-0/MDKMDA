# -*- coding: utf-8 -*-

from keras.layers import *
from keras.layers import Dense
from keras.regularizers import l2
from keras.models import Model
from keras import backend as K  # use computable function
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_recall_curve
import sklearn.metrics as m
from layers import Aggregator
from callbacks import KGCNMetric
from models.base_model import BaseModel
from sklearn.decomposition import PCA
import numpy as np

class KGCN(BaseModel):
    def __init__(self, config):
        super(KGCN, self).__init__(config)

    def build(self):
        input_miRNA = Input(
            shape=(1, ), name='input_miRNA', dtype='int64')
        input_disease = Input(
            shape=(1, ), name='input_disease', dtype='int64')

        miRNA_embedding = Embedding(input_dim=self.config.entity_vocab_size,
                                       output_dim=self.config.embed_dim,
                                       embeddings_initializer='glorot_normal',
                                       embeddings_regularizer=l2(
                                           self.config.l2_weight),
                                       name='miRNA_embedding')(input_miRNA)
        disease_embedding = Embedding(input_dim=self.config.entity_vocab_size,
                                       output_dim=self.config.embed_dim,
                                       embeddings_initializer='glorot_normal',
                                       embeddings_regularizer=l2(
                                           self.config.l2_weight),
                                       name='disease_embedding')(input_disease)
        entity_embedding = Embedding(input_dim=self.config.entity_vocab_size,
                                     output_dim=self.config.embed_dim,
                                     embeddings_initializer='glorot_normal',
                                     embeddings_regularizer=l2(
                                         self.config.l2_weight),
                                     name='entity_embedding')
        relation_embedding = Embedding(input_dim=self.config.relation_vocab_size,
                                       output_dim=self.config.embed_dim,
                                       embeddings_initializer='glorot_normal',
                                       embeddings_regularizer=l2(
                                           self.config.l2_weight),
                                       name='relation_embedding')

        #drug_embed = drug_one_embedding(
        #    input_drug_one)  # [batch_size, 1, embed_dim]

        receptive_list_drug_one = Lambda(lambda x: self.get_receptive_field(x),
                                         name='receptive_filed_drug_one')(input_miRNA)
        neineigh_ent_list_drug_one = receptive_list_drug_one[:self.config.n_depth+1]
        neigh_rel_list_drug_one = receptive_list_drug_one[self.config.n_depth+1:]

        neigh_ent_embed_list_drug_one = [entity_embedding(
            neigh_ent) for neigh_ent in neineigh_ent_list_drug_one]
        neigh_rel_embed_list_drug_one = [relation_embedding(
            neigh_rel) for neigh_rel in neigh_rel_list_drug_one]

        neighbor_embedding = Lambda(lambda x: self.get_neighbor_info(x[0], x[1], x[2]),
                                    name='neighbor_embedding_drug_one')

        for depth in range(self.config.n_depth):
            aggregator = Aggregator[self.config.aggregator_type](
                activation='tanh' if depth == self.config.n_depth-1 else 'relu',
                regularizer=l2(self.config.l2_weight),
                name=f'aggregator_{depth+1}_drug_one'
            )

            next_neigh_ent_embed_list_drug_one = []
            for hop in range(self.config.n_depth-depth):
                neighbor_embed = neighbor_embedding([miRNA_embedding, neigh_rel_embed_list_drug_one[hop],
                                                     neigh_ent_embed_list_drug_one[hop + 1]])
                next_entity_embed = aggregator(
                    [neigh_ent_embed_list_drug_one[hop], neighbor_embed])
                next_neigh_ent_embed_list_drug_one.append(next_entity_embed)
            neigh_ent_embed_list_drug_one = next_neigh_ent_embed_list_drug_one

        # get receptive field
        receptive_list = Lambda(lambda x: self.get_receptive_field(x),
                                name='receptive_filed')(input_disease)
        neigh_ent_list = receptive_list[:self.config.n_depth+1]
        neigh_rel_list = receptive_list[self.config.n_depth+1:]

        neigh_ent_embed_list = [entity_embedding(
            neigh_ent) for neigh_ent in neigh_ent_list]
        neigh_rel_embed_list = [relation_embedding(
            neigh_rel) for neigh_rel in neigh_rel_list]

        neighbor_embedding = Lambda(lambda x: self.get_neighbor_info(x[0], x[1], x[2]),
                                    name='neighbor_embedding')

        for depth in range(self.config.n_depth):
            aggregator = Aggregator[self.config.aggregator_type](
                activation='tanh' if depth == self.config.n_depth-1 else 'relu',
                regularizer=l2(self.config.l2_weight),
                name=f'aggregator_{depth+1}'
            )

            next_neigh_ent_embed_list = []
            for hop in range(self.config.n_depth-depth):
                neighbor_embed = neighbor_embedding([disease_embedding, neigh_rel_embed_list[hop],
                                                     neigh_ent_embed_list[hop + 1]])
                next_entity_embed = aggregator(
                    [neigh_ent_embed_list[hop], neighbor_embed])
                next_neigh_ent_embed_list.append(next_entity_embed)
            neigh_ent_embed_list = next_neigh_ent_embed_list

        miRNA_squeeze_embed = Lambda(lambda x: K.squeeze(x, axis=1))(neigh_ent_embed_list_drug_one[0])
        disease_squeeze_embed = Lambda(lambda x: K.squeeze(x, axis=1))(neigh_ent_embed_list[0])
        print('KGNN_mir_embedding=',miRNA_squeeze_embed)
        print('KGNN_dis_embedding=',disease_squeeze_embed)
        # drug_drug_score = Lambda(
        #     lambda x: K.sigmoid(K.sum(x[0] * x[1], axis=-1, keepdims=True))
        # )([drug1_squeeze_embed, drug2_squeeze_embed])
        

        # 
        miRNA_fusion_Sim_embedding = Lambda(lambda x:self.get_term_miRNA_fusion_embedding(x))(input_miRNA)
        dis_fusion_Sim_embedding = Lambda(lambda x:self.get_term_disease_fusion_embedding(x))(input_disease)
        miRNA_sema_Sim_embedding = Lambda(lambda x:self.get_term_miRNA_sema_embedding(x))(input_miRNA)
        dis_func_Sim_embedding = Lambda(lambda x:self.get_term_disease_func_embedding(x))(input_disease)
        
        miRNA_squeeze_fusion_Sim_embedding = Lambda(lambda x:K.squeeze(x,axis=1))(miRNA_fusion_Sim_embedding)
        dis_squeeze_fusion_Sim_embedding = Lambda(lambda x:K.squeeze(x,axis=1))(dis_fusion_Sim_embedding)
        miRNA_squeeze_sema_Sim_embedding = Lambda(lambda x:K.squeeze(x,axis=1))(miRNA_sema_Sim_embedding)
        dis_squeeze_func_Sim_embedding = Lambda(lambda x:K.squeeze(x,axis=1))(dis_func_Sim_embedding)
        print('miRNA_squeeze_func_Sim_embedding=',miRNA_squeeze_fusion_Sim_embedding)

        # 
        dis_squeeze_HF_embedding = Lambda(lambda x:K.concatenate([x[0],x[1]]))([dis_squeeze_fusion_Sim_embedding,dis_squeeze_func_Sim_embedding])
        mir_squeeze_HF_embedding = Lambda(lambda x:K.concatenate([x[0],x[1]]))([miRNA_squeeze_fusion_Sim_embedding,miRNA_squeeze_sema_Sim_embedding])
        
        # 
        mir_squeeze_embedding = Lambda(lambda x:K.concatenate([x[0],x[1]]))([miRNA_squeeze_embed,mir_squeeze_HF_embedding])
        dis_squeeze_sembedding = Lambda(lambda x:K.concatenate([x[0],x[1]]))([disease_squeeze_embed,dis_squeeze_HF_embedding])
      
        print('final_mir_embedding=',mir_squeeze_embedding)
        print('final_dis_embedding=',dis_squeeze_sembedding)

        final_mir_dis_embedding = Lambda(lambda x:K.concatenate([x[0],x[1]]))([mir_squeeze_embedding,dis_squeeze_sembedding])
        # keras MLP
        x = Dense(64, activation='relu')(final_mir_dis_embedding)
        x2 = Dense(64, activation='relu')(x)
        mir_dis_score = Dense(1, activation='sigmoid')(x2)


        mir_dis_inner = Lambda(lambda x:x[0] * x[1])([miRNA_squeeze_embed, disease_squeeze_embed])
        mir_dis_concat = Lambda(lambda x:K.concatenate([x[0],x[1]]))([mir_squeeze_embedding,dis_squeeze_sembedding])
        x3 = Dense(128, activation='relu')(mir_dis_concat)
        fin_mir_dis_concat = Lambda(lambda x:K.concatenate([x[0],x[1]]))([mir_dis_inner,x3])
        x4 = Dense(64, activation='relu')(fin_mir_dis_concat)
        mir_dis_score = Dense(1, activation='sigmoid')(x4)

        model = Model([input_miRNA, input_disease], mir_dis_score)
        model.compile(optimizer=self.config.optimizer,
                      loss='binary_crossentropy', metrics=['acc'])
        return model

    def get_term_miRNA_fusion_embedding(self, term):
        '''
        Gain pre_embedding for miRNA
        :param term: input_microbe [batch_size,1]
        :return: pre_embedding_tensor,shape [batch_size,1,pre_embedding_dim]
        '''
        pre_miRNA_fusion_embed_matrix = K.variable(self.config.miRNA_pre_fusion_feature,name='pre_miRNA_fusion_embed_matrix',dtype='float32')
        miRNA_pre_embed = K.gather(pre_miRNA_fusion_embed_matrix,K.cast(term,dtype='int64'))
        return miRNA_pre_embed

    def get_term_disease_fusion_embedding(self, term):
        '''
        Gain pre_embedding for disease
        :param term: input_disease, shape [batch_size,1]
        :return: pre_embedding_tensor,shape [batch_size,1,pre_embedding_dim]
        '''
        pre_disease_fusion_embed_matrix = K.variable(self.config.disease_pre_fusion_feature,name='pre_disease_fusion_embed_matrix',dtype='float32')
        disease_pre_embed = K.gather(pre_disease_fusion_embed_matrix,K.cast(term,dtype='int64'))
        return disease_pre_embed

    def get_term_miRNA_sema_embedding(self, term):
        '''
        Gain pre_embedding for miRNA
        :param term: input_microbe [batch_size,1]
        :return: pre_embedding_tensor,shape [batch_size,1,pre_embedding_dim]
        '''
        pre_miRNA_sema_embed_matrix = K.variable(self.config.miRNA_pre_sema_feature,name='pre_miRNA_gaus_embed_matrix',dtype='float32')
        miRNA_pre_embed = K.gather(pre_miRNA_sema_embed_matrix,K.cast(term,dtype='int64'))
        return miRNA_pre_embed

    def get_term_disease_func_embedding(self, term):
        '''
        Gain pre_embedding for disease
        :param term: input_disease, shape [batch_size,1]
        :return: pre_embedding_tensor,shape [batch_size,1,pre_embedding_dim]
        '''
        pre_disease_func_embed_matrix = K.variable(self.config.disease_pre_func_feature,name='pre_disease_gaus_embed_matrix',dtype='float32')
        disease_pre_embed = K.gather(pre_disease_func_embed_matrix,K.cast(term,dtype='int64'))
        return disease_pre_embed

    def get_receptive_field(self, entity):
        """Calculate receptive field for entity using adjacent matrix

        :param entity: a tensor shaped [batch_size, 1]
        :return: a list of tensor: [[batch_size, 1], [batch_size, neighbor_sample_size],
                                   [batch_size, neighbor_sample_size**2], ...]
        """
        neigh_ent_list = [entity]
        neigh_rel_list = []
        adj_entity_matrix = K.variable(
            self.config.adj_entity, name='adj_entity', dtype='int64')
        adj_relation_matrix = K.variable(self.config.adj_relation, name='adj_relation',
                                         dtype='int64')
        n_neighbor = K.shape(adj_entity_matrix)[1]

        for i in range(self.config.n_depth):
            new_neigh_ent = K.gather(adj_entity_matrix, K.cast(
                neigh_ent_list[-1], dtype='int64'))  # cast function used to transform data type
            new_neigh_rel = K.gather(adj_relation_matrix, K.cast(
                neigh_ent_list[-1], dtype='int64'))
            neigh_ent_list.append(
                K.reshape(new_neigh_ent, (-1, n_neighbor ** (i + 1))))
            neigh_rel_list.append(
                K.reshape(new_neigh_rel, (-1, n_neighbor ** (i + 1))))

        return neigh_ent_list + neigh_rel_list

    def get_neighbor_info(self, drug, rel, ent):
        """Get neighbor representation.

        :param user: a tensor shaped [batch_size, 1, embed_dim]
        :param rel: a tensor shaped [batch_size, neighbor_size ** hop, embed_dim]
        :param ent: a tensor shaped [batch_size, neighbor_size ** hop, embed_dim]
        :return: a tensor shaped [batch_size, neighbor_size ** (hop -1), embed_dim]
        """
        # [batch_size, neighbor_size ** hop, 1] drug-entity score
        drug_rel_score = K.sum(drug * rel, axis=-1, keepdims=True)

        # [batch_size, neighbor_size ** hop, embed_dim]
        weighted_ent = drug_rel_score * ent

        # [batch_size, neighbor_size ** (hop-1), neighbor_size, embed_dim]
        weighted_ent = K.reshape(weighted_ent,
                                 (K.shape(weighted_ent)[0], -1,
                                  self.config.neighbor_sample_size, self.config.embed_dim))

        neighbor_embed = K.sum(weighted_ent, axis=2)
        return neighbor_embed

    def add_metrics(self, x_train, y_train, x_valid, y_valid):
        self.callbacks.append(KGCNMetric(x_train, y_train, x_valid, y_valid,
                                         self.config.aggregator_type, self.config.dataset, self.config.K_Fold))

    def fit(self, x_train, y_train, x_valid, y_valid):
        self.callbacks = []
        self.add_metrics(x_train, y_train, x_valid, y_valid)
        self.init_callbacks()

        print('Logging Info - Start training...')
        self.model.fit(x=x_train, y=y_train, batch_size=self.config.batch_size,
                       epochs=self.config.n_epoch, validation_data=(
                           x_valid, y_valid),
                       callbacks=self.callbacks)
        print('Logging Info - training end...')

    def predict(self, x):
        return self.model.predict(x).flatten()

    def score(self, x, y, threshold=0.5):
        y_true = y.flatten()
        y_pred = self.model.predict(x).flatten()
        auc = roc_auc_score(y_true=y_true, y_score=y_pred)
        p, r, t = precision_recall_curve(y_true=y_true, probas_pred=y_pred)
        aupr = m.auc(r, p)
        y_pred = [1 if prob >= threshold else 0 for prob in y_pred]
        acc = accuracy_score(y_true=y_true, y_pred=y_pred)
        f1 = f1_score(y_true=y_true, y_pred=y_pred)

        return auc, acc, f1, aupr
        

