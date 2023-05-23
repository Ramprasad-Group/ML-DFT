import warnings
warnings.filterwarnings('ignore')
import numpy as np

import argparse
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import sys
from inp_params import train_e,new_weights_e,test_e

import time

import tensorflow as tf
import keras.backend as K
from keras.models import Sequential, Model
from keras.layers import Lambda,Input, concatenate,Dense, Activation, Add, ReLU, TimeDistributed, Reshape,Dot,Flatten,Subtract
from keras.models import load_model
from keras.regularizers import l2,l1
from keras.callbacks import EarlyStopping,ModelCheckpoint
from keras.optimizers import SGD,Adam
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.sequence import pad_sequences

import pymatgen
from pymatgen import io
from pymatgen.io.vasp.outputs import Poscar
from pymatgen.io.vasp.outputs import Chgcar
from random import Random
import os
import gc
from operator import itemgetter
import h5py
import glob
import shutil

from keras.backend.tensorflow_backend import set_session
class Input_parameters:
    train_e=train_e    
    new_weights_e=new_weights_e
    test_e=test_e   
    
inp_args=Input_parameters()
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(ROOT_DIR, '../Trained_models/weights_EFP.hdf5')

tfkl = tf.keras.layers
def init_Emod(padding_size):
    def E_model():
        def single_atom_modelC_E():
            model_input=Input(shape=(709,))
            model_out,basisC=Lambda(lambda x: tf.split(x,[700,9],axis=-1))(model_input)
            basisC=Reshape((3,3))(basisC)
            TbasisC=Lambda(lambda x:tf.keras.backend.permute_dimensions(x,(0,2,1)))(basisC)
            model_out=Dense(300,activation='tanh',activity_regularizer=l2(0.1))(model_out)
            model_out=Dense(300,activation='tanh',activity_regularizer=l2(0.1))(model_out)
            model_out=Dense(300,activation='tanh',activity_regularizer=l2(0.1))(model_out)
            model_out=Dense(300,activation='tanh',activity_regularizer=l2(0.1))(model_out)
            model_out=Dense(300,activation='tanh',activity_regularizer=l2(0.1))(model_out)
            model_out=Dense(10,activation='linear')(model_out)
            E,forces,XX,YY,ZZ,XY,YZ,XZ=Lambda(lambda x: tf.split(x,[1,3,1,1,1,1,1,1],axis=-1))(model_out)
            Energy=Lambda(lambda x: tf.math.abs(x))(E)
            forces_out=Dot(axes=(2,1))([basisC,forces])
            model_out=Lambda(lambda x:tf.keras.backend.concatenate((x[0],x[3],x[5],x[3],x[1],x[4],x[5],x[4],x[2]), axis=-1))([XX,YY,ZZ,XY,YZ,XZ])
            model_out=Reshape((3,3))(model_out)
            model_out=Dot(axes=(2,1))([basisC,model_out])
            model_out=Dot(axes=(2,1))([model_out,TbasisC])
            model_out=Flatten()(model_out)
            XX,XY,XZ,YX,YY,YZ,ZX,ZY,ZZ=Lambda(lambda x: tf.split(x,[1,1,1,1,1,1,1,1,1],axis=-1))(model_out)
            model_out=Lambda(lambda x:tf.keras.backend.concatenate((x[0],x[1],x[2],x[3],x[4],x[5],x[6],x[7]), axis=-1))([Energy,forces_out,XX,YY,ZZ,XY,YZ,XZ])
            model=Model(inputs=model_input, outputs=model_out)
            return model
        def single_atom_modelH_E():
            model_input=Input(shape=(577,))
            model_out,basisH=Lambda(lambda x: tf.split(x,[568,9],axis=-1))(model_input)
            basisH=Reshape((3,3))(basisH)
            TbasisH=Lambda(lambda x:tf.keras.backend.permute_dimensions(x,(0,2,1)))(basisH)
            model_out=Dense(300,activation='tanh',activity_regularizer=l2(0.1))(model_out)
            model_out=Dense(300,activation='tanh',activity_regularizer=l2(0.1))(model_out)
            model_out=Dense(300,activation='tanh',activity_regularizer=l2(0.1))(model_out)
            model_out=Dense(300,activation='tanh',activity_regularizer=l2(0.1))(model_out)
            model_out=Dense(300,activation='tanh',activity_regularizer=l2(0.1))(model_out)
            model_out=Dense(10,activation='linear')(model_out)
            E,forces,XX,YY,ZZ,XY,YZ,XZ=Lambda(lambda x: tf.split(x,[1,3,1,1,1,1,1,1],axis=-1))(model_out)
            Energy=Lambda(lambda x: tf.math.abs(x))(E)
            forces_out=Dot(axes=(2,1))([basisH,forces])
            model_out=Lambda(lambda x:tf.keras.backend.concatenate((x[0],x[3],x[5],x[3],x[1],x[4],x[5],x[4],x[2]), axis=-1))([XX,YY,ZZ,XY,YZ,XZ])
            model_out=Reshape((3,3))(model_out)
            model_out=Dot(axes=(2,1))([basisH,model_out])
            model_out=Dot(axes=(2,1))([model_out,TbasisH])
            model_out=Flatten()(model_out)
            XX,XY,XZ,YX,YY,YZ,ZX,ZY,ZZ=Lambda(lambda x: tf.split(x,[1,1,1,1,1,1,1,1,1],axis=-1))(model_out)
            model_out=Lambda(lambda x:tf.keras.backend.concatenate((x[0],x[1],x[2],x[3],x[4],x[5],x[6],x[7]), axis=-1))([Energy,forces_out,XX,YY,ZZ,XY,YZ,XZ])
            model=Model(inputs=model_input, outputs=model_out)
            return model
        def single_atom_modelN_E():
            model_input=Input(shape=(709,))
            model_out,basisC=Lambda(lambda x: tf.split(x,[700,9],axis=-1))(model_input)
            basisC=Reshape((3,3))(basisC)
            TbasisC=Lambda(lambda x:tf.keras.backend.permute_dimensions(x,(0,2,1)))(basisC)
            model_out=Dense(300,activation='tanh',activity_regularizer=l2(0.1))(model_out)
            model_out=Dense(300,activation='tanh',activity_regularizer=l2(0.1))(model_out)
            model_out=Dense(300,activation='tanh',activity_regularizer=l2(0.1))(model_out)
            model_out=Dense(300,activation='tanh',activity_regularizer=l2(0.1))(model_out)
            model_out=Dense(300,activation='tanh',activity_regularizer=l2(0.1))(model_out)
            model_out=Dense(10,activation='linear')(model_out)
            E,forces,XX,YY,ZZ,XY,YZ,XZ=Lambda(lambda x: tf.split(x,[1,3,1,1,1,1,1,1],axis=-1))(model_out)
            Energy=Lambda(lambda x: tf.math.abs(x))(E)
            forces_out=Dot(axes=(2,1))([basisC,forces])
            model_out=Lambda(lambda x:tf.keras.backend.concatenate((x[0],x[3],x[5],x[3],x[1],x[4],x[5],x[4],x[2]), axis=-1))([XX,YY,ZZ,XY,YZ,XZ])
            model_out=Reshape((3,3))(model_out)
            model_out=Dot(axes=(2,1))([basisC,model_out])
            model_out=Dot(axes=(2,1))([model_out,TbasisC])
            model_out=Flatten()(model_out)
            XX,XY,XZ,YX,YY,YZ,ZX,ZY,ZZ=Lambda(lambda x: tf.split(x,[1,1,1,1,1,1,1,1,1],axis=-1))(model_out)
            model_out=Lambda(lambda x:tf.keras.backend.concatenate((x[0],x[1],x[2],x[3],x[4],x[5],x[6],x[7]), axis=-1))([Energy,forces_out,XX,YY,ZZ,XY,YZ,XZ])
            model=Model(inputs=model_input, outputs=model_out)
            return model
        def single_atom_modelO_E():
            model_input=Input(shape=(709,))
            model_out,basisC=Lambda(lambda x: tf.split(x,[700,9],axis=-1))(model_input)
            basisC=Reshape((3,3))(basisC)
            TbasisC=Lambda(lambda x:tf.keras.backend.permute_dimensions(x,(0,2,1)))(basisC)
            model_out=Dense(300,activation='tanh',activity_regularizer=l2(0.1))(model_out)
            model_out=Dense(300,activation='tanh',activity_regularizer=l2(0.1))(model_out)
            model_out=Dense(300,activation='tanh',activity_regularizer=l2(0.1))(model_out)
            model_out=Dense(300,activation='tanh',activity_regularizer=l2(0.1))(model_out)
            model_out=Dense(300,activation='tanh',activity_regularizer=l2(0.1))(model_out)
            model_out=Dense(10,activation='linear')(model_out)
            E,forces,XX,YY,ZZ,XY,YZ,XZ=Lambda(lambda x: tf.split(x,[1,3,1,1,1,1,1,1],axis=-1))(model_out)
            Energy=Lambda(lambda x: tf.math.abs(x))(E)
            forces_out=Dot(axes=(2,1))([basisC,forces])
            model_out=Lambda(lambda x:tf.keras.backend.concatenate((x[0],x[3],x[5],x[3],x[1],x[4],x[5],x[4],x[2]), axis=-1))([XX,YY,ZZ,XY,YZ,XZ])
            model_out=Reshape((3,3))(model_out)
            model_out=Dot(axes=(2,1))([basisC,model_out])
            model_out=Dot(axes=(2,1))([model_out,TbasisC])
            model_out=Flatten()(model_out)
            XX,XY,XZ,YX,YY,YZ,ZX,ZY,ZZ=Lambda(lambda x: tf.split(x,[1,1,1,1,1,1,1,1,1],axis=-1))(model_out)
            model_out=Lambda(lambda x:tf.keras.backend.concatenate((x[0],x[1],x[2],x[3],x[4],x[5],x[6],x[7]), axis=-1))([Energy,forces_out,XX,YY,ZZ,XY,YZ,XZ])
            model=Model(inputs=model_input, outputs=model_out)
            return model
        def se(y_true, y_pred):
                mask = K.all(K.equal(y_true, 1000), axis=-1,keepdims=True)
                mask = 1 - K.cast(mask, K.floatx())
                diff=Subtract()([y_true,y_pred])
                loss = Lambda(lambda x: tf.square(x))(diff)
                loss = Lambda(lambda x: x[0]*x[1])([loss,mask])
                loss=Lambda(lambda x: tf.keras.backend.sum(x))(loss)
                cases=Lambda(lambda x: tf.keras.backend.sum(x))(mask)
                loss=Lambda(lambda x: x[0]/x[1])([loss,cases])
                return loss
        input1=Input(shape=(padding_size,709))
        input2=Input(shape=(padding_size,577))
        input3=Input(shape=(padding_size,709))
        input4=Input(shape=(padding_size,709))
        input5=Input(shape=(1,))
        input6=Input(shape=(padding_size,1))
        input7=Input(shape=(padding_size,1))
        input8=Input(shape=(padding_size,1))
        input9=Input(shape=(padding_size,1))
        model_out_CP=TimeDistributed(single_atom_modelC_E(),name='atom_C_P')(input1)
        model_out_HP=TimeDistributed(single_atom_modelH_E(),name='atom_H_P')(input2)
        model_out_NP=TimeDistributed(single_atom_modelN_E(),name='atom_N_P')(input3)
        model_out_OP=TimeDistributed(single_atom_modelO_E(),name='atom_O_P')(input4)
        EC,forcesC,pressC=Lambda(lambda x: tf.split(x,[1,3,6],axis=-1))(model_out_CP)
        EH,forcesH,pressH=Lambda(lambda x: tf.split(x,[1,3,6],axis=-1))(model_out_HP)
        EN,forcesN,pressN=Lambda(lambda x: tf.split(x,[1,3,6],axis=-1))(model_out_NP)
        EO,forcesO,pressO=Lambda(lambda x: tf.split(x,[1,3,6],axis=-1))(model_out_OP)
        model_added=Add()([pressC, pressH, pressN,pressO])
        EC=Lambda(lambda x: tf.math.multiply(x[0],x[1]),name='new_EC')([input6,EC])
        EH=Lambda(lambda x: tf.math.multiply(x[0],x[1]),name='new_EH')([input7,EH])
        EN=Lambda(lambda x: tf.math.multiply(x[0],x[1]),name='new_EN')([input8,EN])
        EO=Lambda(lambda x: tf.math.multiply(x[0],x[1]),name='new_EO')([input9,EO])
        E_added=Add()([EC, EH,EN,EO])
        E_tot=Lambda(lambda x: tf.keras.backend.sum(x,axis=1))(E_added)
        E_tot=Lambda(lambda x: x/input5,name='Energy')(E_tot)
        model_p=Lambda(lambda x: tf.keras.backend.sum(x,axis=1))(model_added)
        model_p=Lambda(lambda x: x/input5,name='Press')(model_p)
#
        model= Model(inputs=[input1,input2,input3,input4,input5,input6,input7,input8,input9], outputs=[E_tot,forcesC,forcesH,forcesN,forcesO,model_p])

        opt = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999)
        model.compile(loss=["mean_squared_error",se,se,se,se,"mean_squared_error"],optimizer=opt,loss_weights=[1000,10,10,10,10,0.1])
#
        return model
    model_E=E_model()
    return model_E

def model_weights(train_e,new_weights_e,model_E):
    if train_e:
        model_E.load_weights('newEmodel.hdf5')
    elif new_weights_e:
        model_E.load_weights('newEmodel.hdf5')
    else:
        model_E.load_weights(CONFIG_PATH)


def energy_predict(X_C,X_H,X_N,X_O,basis1,basis2,basis3,basis4,C_m,H_m,N_m,O_m,num_atoms,model_E):
    X_C=np.concatenate((X_C,basis1), axis=-1)
    X_H=np.concatenate((X_H,basis2), axis=-1)
    X_N=np.concatenate((X_N,basis3), axis=-1)
    X_O=np.concatenate((X_O,basis4), axis=-1)
    model_weights(train_e,new_weights_e,model_E)
    E,ForC,ForH,ForN,ForO,pred_press=model_E.predict([X_C,X_H,X_N,X_O,num_atoms,C_m,H_m,N_m,O_m], batch_size=1)
    Pred_Energy=(-1)*E
    ForC=np.squeeze(ForC)
    ForH=np.squeeze(ForH)
    ForN=np.squeeze(ForN)
    ForO=np.squeeze(ForO)
    pred_press=np.squeeze(pred_press)
    return Pred_Energy[0][0],ForC,ForH,ForN,ForO,pred_press

def e_train(file_loc,tot_atoms):
    levels_file=os.path.join(file_loc,"energy")
    levels_data=[[float(s) for s in l.split()] for l in open(levels_file).readlines()]
    Energy=abs(np.array(levels_data[0]))/tot_atoms
    forces_file=os.path.join(file_loc,"forces")
    forces_data=np.array([[float(s) for s in l.split()] for l in open(forces_file).readlines()])
    press_file=os.path.join(file_loc,"stress")
    press_data=np.array([[float(s) for s in l.split()] for l in open(press_file).readlines()])
    press_data=np.reshape(press_data,(6))
    press=np.reshape(press_data,(1,6))

    return Energy,forces_data,press

def retrain_emodel(X_C,X_H,X_N,X_O,C_m,H_m,N_m,O_m,basis1,basis2,basis3,basis4,X_at,ener_ref,forces1,forces2,forces3,forces4,press_ref,X_val_C,X_val_H,X_val_N,X_val_O,C_mV,H_mV,N_mV,O_mV,basis1V,basis2V,basis3V,basis4V,X_at_val,ener_val,forces1V,forces2V,forces3V,forces4V,press_val,ert_epochs,ert_batch_size,ert_patience,padding_size):
    X_C=np.concatenate((X_C,basis1), axis=-1)
    X_H=np.concatenate((X_H,basis2), axis=-1)
    X_N=np.concatenate((X_N,basis3), axis=-1)
    X_O=np.concatenate((X_O,basis4), axis=-1)
    X_val_C=np.concatenate((X_val_C,basis1V), axis=-1)
    X_val_H=np.concatenate((X_val_H,basis2V), axis=-1)
    X_val_N=np.concatenate((X_val_N,basis3V), axis=-1)
    X_val_O=np.concatenate((X_val_O,basis4V), axis=-1)
    filepath="newEmodel.hdf5"
    rtmodel = init_Emod(padding_size)
    rtmodel.load_weights(CONFIG_PATH)
    checkpoint=ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=True,mode='min')
    early_stopping_cb = EarlyStopping(patience=ert_patience,restore_best_weights=True)
    callbacks_list = [checkpoint,early_stopping_cb]

    history=rtmodel.fit([X_C,X_H,X_N,X_O,X_at,C_m,H_m,N_m,O_m],[ener_ref,forces1,forces2,forces3,forces4,press_ref],epochs=ert_epochs, batch_size=ert_batch_size,shuffle=True,validation_data=([X_val_C,X_val_H,X_val_N,X_val_O,X_at_val,C_mV,H_mV,N_mV,O_mV],[ener_val,forces1V,forces2V,forces3V,forces4V,press_val]),callbacks=callbacks_list)
