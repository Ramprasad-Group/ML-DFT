import warnings
warnings.filterwarnings('ignore')
import numpy as np
import argparse
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import sys
from inp_params import train_dos,new_weights_dos
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score

import time

import tensorflow as tf
import keras.backend as K
from keras.models import Model
from keras.layers import Lambda,Input, concatenate,Dense,Reshape, Conv1D,Activation, Add, ReLU, TimeDistributed,Dropout
from keras.models import load_model
from keras.regularizers import l2,l1
from keras.callbacks import EarlyStopping,ModelCheckpoint
from keras.optimizers import SGD,Adam
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.sequence import pad_sequences

import pymatgen
from pymatgen import io
from pymatgen.io.vasp.outputs import Poscar


import random

import gc
from operator import itemgetter
import h5py
from numpy import asarray
import h5py
import glob
import shutil
from keras.backend.tensorflow_backend import set_session
class Input_parameters:
    train_dos=train_dos
    new_weights_dos=new_weights_dos

inp_args=Input_parameters()

tfkl = tf.keras.layers

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH1 = os.path.join(ROOT_DIR, '../Trained_models/weights_DOS.hdf5')

def init_DOSmod(padding_size):
    def dos_model():
        def single_atom_modelC_1():
            model_input=Input(shape=(700,))
            model_out=Dense(600,activation='relu',activity_regularizer=l2(0.1),kernel_initializer='glorot_uniform')(model_input)
            model_out=Dropout(0.1)(model_out,training=True)
            model_out=Dense(600,activation='relu',activity_regularizer=l2(0.1),kernel_initializer='glorot_uniform')(model_out)
            model_out=Dropout(0.1)(model_out,training=True)
            model_out=Dense(600,activation='relu',activity_regularizer=l2(0.1),kernel_initializer='glorot_uniform')(model_out)
            model_out=Dropout(0.1)(model_out,training=True)
            model_out=Dense(600,activation='relu',activity_regularizer=l2(0.1),kernel_initializer='glorot_uniform')(model_out)
            model_out=Dropout(0.1)(model_out,training=True)
            model_out=Dense(343,activation='relu',activity_regularizer=l2(0.1),kernel_initializer='glorot_uniform')(model_out)
            model_out=Reshape((343,-1))(model_out)
            model_out=Conv1D(3,kernel_size=3,activation='relu')(model_out)
            model_out=Lambda(lambda x: tf.keras.backend.mean(x,axis=-1))(model_out)
            model=Model(inputs=model_input, outputs=model_out)
#            model.summary()
            return model
        def single_atom_modelH_1():
            model_input=Input(shape=(568,))
            model_out=Dense(600,activation='relu',activity_regularizer=l2(0.1),kernel_initializer='glorot_uniform')(model_input)
            model_out=Dropout(0.1)(model_out,training=True)
            model_out=Dense(600,activation='relu',activity_regularizer=l2(0.1),kernel_initializer='glorot_uniform')(model_out)
            model_out=Dropout(0.1)(model_out,training=True)
            model_out=Dense(600,activation='relu',activity_regularizer=l2(0.1),kernel_initializer='glorot_uniform')(model_out)
            model_out=Dropout(0.1)(model_out,training=True)
            model_out=Dense(600,activation='relu',activity_regularizer=l2(0.1),kernel_initializer='glorot_uniform')(model_out)
            model_out=Dropout(0.1)(model_out,training=True)
            model_out=Dense(343,activation='relu',activity_regularizer=l2(0.1),kernel_initializer='glorot_uniform')(model_out)
            model_out=Reshape((343,-1))(model_out)
            model_out=Conv1D(3,kernel_size=3,activation='relu')(model_out)
            model_out=Lambda(lambda x: tf.keras.backend.mean(x,axis=-1))(model_out)
            model=Model(inputs=model_input, outputs=model_out)
            return model
        def single_atom_modelN_1():
            model_input=Input(shape=(700,))
            model_out=Dense(600,activation='relu',activity_regularizer=l2(0.1),kernel_initializer='glorot_uniform')(model_input)
            model_out=Dropout(0.1)(model_out,training=True)
            model_out=Dense(600,activation='relu',activity_regularizer=l2(0.1),kernel_initializer='glorot_uniform')(model_out)
            model_out=Dropout(0.1)(model_out,training=True)
            model_out=Dense(600,activation='relu',activity_regularizer=l2(0.1),kernel_initializer='glorot_uniform')(model_out)
            model_out=Dropout(0.1)(model_out,training=True)
            model_out=Dense(600,activation='relu',activity_regularizer=l2(0.1),kernel_initializer='glorot_uniform')(model_out)
            model_out=Dropout(0.1)(model_out,training=True)
            model_out=Dense(343,activation='relu',activity_regularizer=l2(0.1),kernel_initializer='glorot_uniform')(model_out)
            model_out=Reshape((343,-1))(model_out)
            model_out=Conv1D(3,kernel_size=3,activation='relu')(model_out)
            model_out=Lambda(lambda x: tf.keras.backend.mean(x,axis=-1))(model_out)
            model=Model(inputs=model_input, outputs=model_out)
            return model
        def single_atom_modelO_1():
            model_input=Input(shape=(700,))
            model_out=Dense(600,activation='relu',activity_regularizer=l2(0.1),kernel_initializer='glorot_uniform')(model_input)
            model_out=Dropout(0.1)(model_out,training=True)
            model_out=Dense(600,activation='relu',activity_regularizer=l2(0.1),kernel_initializer='glorot_uniform')(model_out)
            model_out=Dropout(0.1)(model_out,training=True)
            model_out=Dense(600,activation='relu',activity_regularizer=l2(0.1),kernel_initializer='glorot_uniform')(model_out)
            model_out=Dropout(0.1)(model_out,training=True)
            model_out=Dense(600,activation='relu',activity_regularizer=l2(0.1),kernel_initializer='glorot_uniform')(model_out)
            model_out=Dropout(0.1)(model_out,training=True)
            model_out=Dense(343,activation='relu',activity_regularizer=l2(0.1),kernel_initializer='glorot_uniform')(model_out)
            model_out=Reshape((343,-1))(model_out)
            model_out=Conv1D(3,kernel_size=3,activation='relu')(model_out)
            model_out=Lambda(lambda x: tf.keras.backend.mean(x,axis=-1))(model_out)
            model=Model(inputs=model_input, outputs=model_out)
            return model
        input1=Input(shape=(padding_size,700))
        input2=Input(shape=(padding_size,568))
        input3=Input(shape=(padding_size,700))
        input4=Input(shape=(padding_size,700))
        input5=Input(shape=(1,))
        model_out_C1=TimeDistributed(single_atom_modelC_1(),name='atom_dosC_1')(input1)
        model_out_H1=TimeDistributed(single_atom_modelH_1(),name='atom_dosH_1')(input2)
        model_out_N1=TimeDistributed(single_atom_modelN_1(),name='atom_dosN_1')(input3)
        model_out_O1=TimeDistributed(single_atom_modelO_1(),name='atom_dosO_1')(input4)
        model_added1=Add()([model_out_C1, model_out_H1,model_out_N1,model_out_O1])

        model_s1=Lambda(lambda x: tf.keras.backend.sum(x,axis=1))(model_added1)
        model_dos=Lambda(lambda x: x/input5,name='DOS1')(model_s1)
        bands=Dense(100,activation='relu',activity_regularizer=l2(0.01))(model_dos)
        bands=Dense(100,activation='relu',activity_regularizer=l2(0.01))(bands)
        bands=Dense(2,activation='relu')(bands)
        model= Model(inputs=[input1,input2,input3,input4,input5], outputs=[model_dos,bands])
        opt = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999)
        model.compile(loss="mean_squared_error",optimizer=opt,loss_weights=[1000,1])

        return model
    modelDOS=dos_model()
    return modelDOS

def Dmodel_weights(train_dos,new_weights_dos,modelDOS):
    if train_dos:
        modelDOS.load_weights('newDOSmodel.hdf5')
    elif new_weights_dos:
        modelDOS.load_weights('newDOSmodel.hdf5')
    else:
        modelDOS.load_weights(CONFIG_PATH1)


def DOS_pred(X_C,X_H,X_N,X_O,total_elec,modelDOS):
    resultD = []
    resultvbcb=[]
    Dmodel_weights(train_dos,new_weights_dos,modelDOS)
    for i in range(100):
        Pred,vbcb=modelDOS.predict([X_C,X_H,X_N,X_O,total_elec], batch_size=1)
        resultD.append(Pred*total_elec)
        resultvbcb.append(vbcb)
    resultD=np.array(resultD)
    Pred = resultD.mean(axis=0)
    resultvbcb=np.array(resultvbcb)
    devVB=resultvbcb.std(axis=0)[0][0]
    devCB=resultvbcb.std(axis=0)[0][1]
    Pred_vb =(-1)*resultvbcb.mean(axis=0)[0][0]
    Pred_cb =(-1)*resultvbcb.mean(axis=0)[0][1]
    uncertainty = resultD.std(axis=0)
    uncertainty=np.squeeze(uncertainty)
    resultvbcb=np.vstack(resultvbcb)
    Bandgap=Pred_cb-Pred_vb
    devBG=resultvbcb[:,0]-resultvbcb[:,1]
    devBG=devBG.std(axis=0)
    return Pred, uncertainty,Pred_vb,devVB,Pred_cb,devCB,Bandgap,devBG


def DOS_plot(energy_wind,Pred,VB,CB,uncertainty, localfile_loc):
    plt.plot(energy_wind, Pred,"r-", label='DOS',  linewidth=1)
    plt.fill_between(energy_wind,Pred-uncertainty,Pred+uncertainty, color='gray', alpha=0.2)
    plt.axvline(VB, color='b', label='Valence band', linestyle=':')
    plt.axvline(CB, color='g', label='Conduction band', linewidth=1)
    plt.axvline(0, color='k', label='Vacuum level', linestyle='dashed', linewidth=2)
    plt.legend(fontsize=16)
    plt.tick_params(labelsize=16)
    plt.xlabel("Energy (eV)", fontsize=18)
    plt.ylabel("DOS", fontsize=18)
    plt.tight_layout()
    plt.savefig("dos_" + localfile_loc + " .png", dpi=500)
    plt.clf()
    print("Made the dos plot ..")

def dos_data(file_loc,total_elec):
    dos_file = os.path.join(file_loc,"dos")
    dos_data=[[float(s) for s in l.split()] for l in open(dos_file).readlines()]
    dos=[]
    for i in range(len(dos_data)):
        dos.append(dos_data[i][0])
    levels_file=os.path.join(file_loc,"VB_CB")
    levels_data=[[float(s) for s in l.split()] for l in open(levels_file).readlines()]
    VB=abs(np.array(levels_data[0]))
    CB=abs(np.array(levels_data[1]))
    Prop=np.array(dos)/total_elec
    return Prop, VB, CB

def retrain_dosmodel(X_C,X_H,X_N,X_O,X_el,Prop_dos,vbcb,X_val_C,X_val_H,X_val_N,X_val_O,X_el_val,Prop_dos_val,vbcb_val,drt_epochs,drt_batch_size, drt_patience,padding_size):
    filepath="newDOSmodel.hdf5"
    rtmodel = init_DOSmod(padding_size)
    rtmodel.load_weights(CONFIG_PATH1)
    checkpoint=ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=True,mode='min')
    early_stopping_cb = EarlyStopping(patience=drt_patience,restore_best_weights=True)
    callbacks_list = [checkpoint,early_stopping_cb]
    history=rtmodel.fit([X_C,X_H,X_N,X_O,X_el],[Prop_dos,vbcb],epochs=drt_epochs, batch_size=drt_batch_size,shuffle=True,validation_data=([X_val_C,X_val_H,X_val_N,X_val_O,X_el_val],[Prop_dos_val,vbcb_val]),callbacks=callbacks_list)
 
