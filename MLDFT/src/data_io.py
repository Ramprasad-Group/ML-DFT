import warnings
warnings.filterwarnings('ignore')
import numpy as np
from numpy import cumsum

import argparse
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import sys
sys.path.append(os.getcwd())

import tensorflow as tf
import keras.backend as K
from keras.preprocessing.sequence import pad_sequences

import pymatgen
from pymatgen import io
from pymatgen.io.vasp.outputs import Poscar

from random import Random

from operator import itemgetter
import h5py
import pathlib
import glob
import shutil

from keras.backend.tensorflow_backend import set_session
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
from MLDFT.src.FP import fp_atom
from MLDFT.src.CHG import chg_train,chg_dat_prep,coef_predict
from MLDFT.src.Energy import e_train
from MLDFT.src.DOS import DOS_pred,dos_data
elec_dict={6:4,  1:1, 7:5,8:6}

tfkl = tf.keras.layers

def get_def_data(file_loc):
    print(file_loc)
    poscar_file = os.path.join(file_loc,"POSCAR")
    poscar_data=Poscar.from_file(poscar_file)
    vol = poscar_data.structure.volume
    supercell = poscar_data.structure
    dim=supercell.lattice.matrix
    atoms=supercell.num_sites
    elems_list = sorted(list(set(poscar_data.site_symbols)))
    print('elem_list', elems_list)
    electrons_list = [elec_dict[x] for x in list(poscar_data.structure.atomic_numbers)]
    total_elec = sum(electrons_list)
    print('total_elec', total_elec)
    return vol, supercell, dim, total_elec,elems_list,poscar_data

def get_fp_all(at_elem,X_tot,padding_size):
    X_npad=[]
    h=0
    pp=1
    i1=at_elem[0]
    i2=at_elem[1]
    i3=at_elem[2]
    i4=at_elem[3]
    for i in [at_elem[0],at_elem[1]]:
        f=X_tot[h:i+h,0:360]
        f_pad=pad_sequences(f.T, maxlen=padding_size,dtype='float32',padding='post',value=0.0)
        X_npad.append(f_pad)
        h=h+i
        pp=pp+1

    if at_elem[2] != 0:
        f=X_tot[i1+i2:i1+i2+i3,0:360]
        f_pad=pad_sequences(f.T, maxlen=padding_size,dtype='float32',padding='post',value=0.0)
        X_npad.append(f_pad)

    else:
        f=[0]*360
        f=np.array(np.reshape(f,(1,360)))
        f_pad=pad_sequences(f.T, maxlen=padding_size,dtype='float32',padding='post',value=0.0)
        X_npad.append(f_pad)

    if at_elem[3] != 0:
        f=X_tot[i1+i2+i3:i1+i2+i3+i4,0:360]
        f_pad=pad_sequences(f.T, maxlen=padding_size,dtype='float32',padding='post',value=0.0)
        X_npad.append(f_pad)
        
    else:
        f=[0]*360
        f=np.array(np.reshape(f,(1,360)))
        f_pad=pad_sequences(f.T, maxlen=padding_size,dtype='float32',padding='post',value=0.0)
        X_npad.append(f_pad)
        
    X_pad=np.concatenate(X_npad,axis=1)
    return X_pad

def get_fp_basis_F(at_elem,X_tot,forces_data,base_mat,padding_size):
    X_npad=[]
    basis_npad=[]
    forces_npad=[]
    h=0
    pp=1
    i1=at_elem[0]
    i2=at_elem[1]
    i3=at_elem[2]
    i4=at_elem[3]
    for i in [at_elem[0],at_elem[1]]:
        f=X_tot[h:i+h,0:360]
        forces=forces_data[h:i+h]
        basis=base_mat[h:i+h]
        basis=np.reshape(basis,(i,9))
        f_pad=pad_sequences(f.T, maxlen=padding_size,dtype='float32',padding='post',value=0.0)
        basis_pad=pad_sequences(basis.T, maxlen=padding_size,dtype='float32',padding='post',value=0.0)
        forces_pad=pad_sequences(forces.T, maxlen=padding_size,dtype='float32',padding='post',value=1000)
        X_npad.append(f_pad)
        basis_npad.append(basis_pad)
        forces_npad.append(forces_pad)

        h=h+i
        pp=pp+1

    if at_elem[2] != 0:
        f=X_tot[i1+i2:i1+i2+i3,0:360]
        forces=forces_data[i1+i2:i1+i2+i3]
        basis=base_mat[i1+i2:i1+i2+i3]
        basis=np.reshape(basis,(i3,9))
        f_pad=pad_sequences(f.T, maxlen=padding_size,dtype='float32',padding='post',value=0.0)
        basis_pad=pad_sequences(basis.T, maxlen=padding_size,dtype='float32',padding='post',value=0.0)
        forces_pad=pad_sequences(forces.T, maxlen=padding_size,dtype='float32',padding='post',value=1000)
        X_npad.append(f_pad)
        basis_npad.append(basis_pad)
        forces_npad.append(forces_pad)

    else:
        f=[0]*360
        f=np.array(np.reshape(f,(1,360)))
        forces=[0,0,0]
        forces=np.array(np.reshape(forces,(1,3)))
        basis=[0,0,0,0,0,0,0,0,0]
        basis=np.array(np.reshape(basis,(1,9)))
        f_pad=pad_sequences(f.T, maxlen=padding_size,dtype='float32',padding='post',value=0.0)
        basis_pad=pad_sequences(basis.T, maxlen=padding_size,dtype='float32',padding='post',value=0.0)
        forces_pad=pad_sequences(forces.T, maxlen=padding_size,dtype='float32',padding='post',value=1000)
        X_npad.append(f_pad)
        basis_npad.append(basis_pad)
        forces_npad.append(forces_pad)
    if at_elem[3] != 0:
        f=X_tot[i1+i2+i3:i1+i2+i3+i4,0:360]
        forces=forces_data[i1+i2+i3:i1+i2+i3+i4]
        basis=base_mat[i1+i2+i3:i1+i2+i3+i4]
        basis=np.reshape(basis,(i4,9))
        f_pad=pad_sequences(f.T, maxlen=padding_size,dtype='float32',padding='post',value=0.0)
        basis_pad=pad_sequences(basis.T, maxlen=padding_size,dtype='float32',padding='post',value=0.0)
        forces_pad=pad_sequences(forces.T, maxlen=padding_size,dtype='float32',padding='post',value=1000)
        X_npad.append(f_pad)
        basis_npad.append(basis_pad)
        forces_npad.append(forces_pad)
    else:
        f=[0]*360
        f=np.array(np.reshape(f,(1,360)))
        forces=[0,0,0]
        forces=np.array(np.reshape(forces,(1,3)))
        basis=[0,0,0,0,0,0,0,0,0]
        basis=np.array(np.reshape(basis,(1,9)))
        f_pad=pad_sequences(f.T, maxlen=padding_size,dtype='float32',padding='post',value=0.0)
        basis_pad=pad_sequences(basis.T, maxlen=padding_size,dtype='float32',padding='post',value=0.0)
        forces_pad=pad_sequences(forces.T, maxlen=padding_size,dtype='float32',padding='post',value=1000)
        X_npad.append(f_pad)
        basis_npad.append(basis_pad)
        forces_npad.append(forces_pad)

    X_pad=np.concatenate(X_npad,axis=1)
    basis_pad=np.concatenate(basis_npad,axis=1)
    forces_pad=np.concatenate(forces_npad,axis=1)

    return X_pad,basis_pad,forces_pad

def get_all_data(data_list):
    X_list_at1=[]
    X_list_at2=[]
    X_list_at3=[]
    X_list_at4=[]
    Prop_list=[]
    Prop_e_list=[]
    Prop_dos_list=[]
    Prop_vbcb_list=[]
    dataset_at1=[]
    dataset_at2=[]
    dataset_at3=[]
    dataset_at4=[]
    At_list=[]
    El_list=[]
    for file_loc in data_list:
        vol,supercell,dim,total_elec,elems_list,poscar_data=get_def_data(file_loc)
        El_list.append(total_elec)
        dset,sites_elem,num_atoms,at_elem=fp_atom(poscar_data,supercell,elems_list)
        At_list.append(num_atoms)
        dataset1=dset[:]
        i1=at_elem[0]
        i2=at_elem[1]
        i3=at_elem[2]
        i4=at_elem[3]
        X_at1=dataset1[0:i1]
        X_at2=dataset1[i1:i1+i2]
        X_at3=dataset1[i1+i2:i1+i2+i3]
        X_at4=dataset1[i1+i2+i3:i1+i2+i3+i4]
        dataset_at1.append(X_at1)
        dataset_at2.append(X_at2)
        dataset_at3.append(X_at3)
        dataset_at4.append(X_at4)
        chg,local_coords=chg_train(file_loc,vol, supercell,sites_elem,num_atoms,at_elem)
        num_chg_bins=chg.shape[0]
        dataset2=local_coords[:]
        X_tot_at1,X_tot_at2,X_tot_at3,X_tot_at4=chg_dat_prep(at_elem,dataset1,dataset2,i1,i2,i3,i4,num_chg_bins)
        Prop_tot=np.array(chg)
        X_list_at1.append(X_tot_at1.T)
        X_list_at2.append(X_tot_at2.T)
        X_list_at3.append(X_tot_at3.T)
        X_list_at4.append(X_tot_at4.T)
        Prop_list.append(Prop_tot)
    X_1 = X_list_at1
    X_2 = X_list_at2
    X_3 = X_list_at3
    X_4 = X_list_at4
    Prop = np.vstack(Prop_list)
    X_at=np.vstack(At_list)
    X_el=np.vstack(El_list)
    return X_1,X_2,X_3,X_4,Prop,dataset_at1,dataset_at2,dataset_at3,dataset_at4,X_at,X_el

def get_efp_data(data_list):
    ener_list=[]
    forces_pre_list=[]
    press_list=[]
    X_pre_list=[]
    basis_pre_list=[]
    At_list=[]
    El_list=[]
    X_at_elem=[]
    for file_loc in data_list:
        vol,supercell,dim,total_elec,elems_list,poscar_data=get_def_data(file_loc)
        El_list.append(total_elec)
        dset,basis_mat,sites_elem,num_atoms,at_elem=fp_atom(poscar_data,supercell,elems_list)
        At_list.append(num_atoms)
        dataset1=dset[:]
        X_pre_list.append(dataset1)
        basis_pre_list.append(basis_mat)
        Prop_e,forces_data,press=e_train(file_loc, num_atoms)
        press_list.append(press)
        ener_list.append(Prop_e)
        forces_pre_list.append(forces_data)
        X_at_elem.append(at_elem)
    X_elem=np.vstack(X_at_elem)
    X_at=np.vstack(At_list)
    X_el=np.vstack(El_list)
    press_ref=np.vstack(press_list)
    ener_ref=np.vstack(ener_list)
    return ener_ref,forces_pre_list,press_ref,X_pre_list,basis_pre_list,X_at,X_el,X_at_elem

def pad_dat(X_at_elem,X_pre_list,padding_size):
    X_list=[]
    for at_elem,dataset1 in zip(X_at_elem,X_pre_list):
        X_pad=get_fp_all(at_elem,dataset1,padding_size)
        X_list.append(X_pad.T)
    X=np.vstack(X_list)
    tot_conf=int(X.shape[0]/(4*padding_size))
    X_3D = np.reshape(X, (tot_conf,4, padding_size, X.shape[1]))
    X_1=X_3D[:,0,:,:]
    X_2=X_3D[:,1,:,:]
    X_3=X_3D[:,2,:,:]
    X_4=X_3D[:,3,:,:]
    return X_1,X_2,X_3,X_4


def pad_efp_data(X_at_elem,X_pre_list,forces_pre_list,basis_pre_list,padding_size):
    X_list=[]
    basis_list=[]
    forces_list=[]
    for at_elem,dataset1,forces_data,basis_mat in zip(X_at_elem,X_pre_list,forces_pre_list,basis_pre_list):
        X_pad,basis_pad,forces_pad=get_fp_basis_F(at_elem,dataset1,forces_data,basis_mat,padding_size)
        X_list.append(X_pad.T)
        basis_list.append(basis_pad.T)
        forces_list.append(forces_pad.T)
    X=np.vstack(X_list)
    basis_ref=np.vstack(basis_list)
    force_ref=np.vstack(forces_list)
    tot_conf=int(X.shape[0]/(4*padding_size))
    X_3D = np.reshape(X, (tot_conf,4, padding_size, X.shape[1]))
    basis_3D=np.reshape(basis_ref, (tot_conf,4, padding_size, 9))
    forces_3D=np.reshape(force_ref,(tot_conf,4,padding_size,3))
    X_1=X_3D[:,0,:,:]
    X_2=X_3D[:,1,:,:]
    X_3=X_3D[:,2,:,:]
    X_4=X_3D[:,3,:,:]
    basis1=basis_3D[:,0,:,:]
    basis2=basis_3D[:,1,:,:]
    basis3=basis_3D[:,2,:,:]
    basis4=basis_3D[:,3,:,:]
    forces1=forces_3D[:,0,:,:]
    forces2=forces_3D[:,1,:,:]
    forces3=forces_3D[:,2,:,:]
    forces4=forces_3D[:,3,:,:]
    return forces1,forces2,forces3,forces4,X_1,X_2,X_3,X_4,basis1,basis2,basis3,basis4

def pad_dos_dat(Prop_vbcb,X_1):
    tot_conf=int(X_1.shape[0])
    Prop_B=np.reshape(Prop_vbcb,(tot_conf,2))
    return Prop_B

def get_e_dos_data(data_list):
    Prop_dos_list=[]
    Prop_vbcb_list=[]
    ener_list=[]
    forces_pre_list=[]
    press_list=[]
    X_pre_list=[]
    basis_pre_list=[]
    At_list=[]
    El_list=[]
    X_at_elem=[]
    for file_loc in data_list:
        vol,supercell,dim,total_elec,elems_list,poscar_data=get_def_data(file_loc)
        El_list.append(total_elec)
        dset,basis_mat,sites_elem,num_atoms,at_elem=fp_atom(poscar_data,supercell,elems_list)
        At_list.append(num_atoms)
        dataset1=dset[:]
        X_pre_list.append(dataset1)
        basis_pre_list.append(basis_mat)
        Prop_e,forces_data,press=e_train(file_loc, num_atoms)
        dos_dat,VB,CB=dos_data(file_loc,total_elec)
        press_list.append(press)
        ener_list.append(Prop_e)
        forces_pre_list.append(forces_data)
        X_at_elem.append(at_elem)
        Prop_dos_list.append(dos_dat)
        Prop_vbcb_list.append(VB)
        Prop_vbcb_list.append(CB)
    X_elem=np.vstack(X_at_elem)
    X_at=np.vstack(At_list)
    X_el=np.vstack(El_list)
    Prop_dos=np.vstack(Prop_dos_list)
    Prop_vbcb=np.vstack(Prop_vbcb_list)
    press_ref=np.vstack(press_list)
    ener_ref=np.vstack(ener_list)
    return ener_ref,forces_pre_list,press_ref,X_pre_list,basis_pre_list,X_at,X_el,X_at_elem,Prop_dos,Prop_vbcb

def get_dos_data(data_list):
    Prop_dos_list=[]
    Prop_vbcb_list=[]
    X_pre_list=[]
    At_list=[]
    El_list=[]
    X_at_elem=[]
    for file_loc in data_list:
        vol,supercell,dim,total_elec,elems_list,poscar_data=get_def_data(file_loc)
        El_list.append(total_elec)
        dset,basis_mat,sites_elem,num_atoms,at_elem=fp_atom(poscar_data,supercell,elems_list)
        At_list.append(num_atoms)
        dataset1=dset[:]
        X_pre_list.append(dataset1)
        X_at_elem.append(at_elem)
        dos_dat,VB,CB=dos_data(file_loc,total_elec)
        Prop_dos_list.append(dos_dat)
        Prop_vbcb_list.append(VB)
        Prop_vbcb_list.append(CB)
    X_elem=np.vstack(X_at_elem)
    X_at=np.vstack(At_list)
    X_el=np.vstack(El_list)
    Prop_dos=np.vstack(Prop_dos_list)
    Prop_vbcb=np.vstack(Prop_vbcb_list)
    return X_pre_list,X_at,X_el,X_at_elem,Prop_dos,Prop_vbcb

def get_dos_e_train_data(X_1,X_2,X_3,X_4,X_elem,padding_size,modelCHG):
    X_C=[]
    X_H=[]
    X_N=[]
    X_O=[]
    for i in range (0,X_1.shape[0]):
        predCHG1,predCHG2,predCHG3,predCHG4=coef_predict(X_1[i].reshape(1,padding_size,360),X_2[i].reshape(1,padding_size,360),X_3[i].reshape(1,padding_size,360),X_4[i].reshape(1,padding_size,360),X_elem[i][0],X_elem[i][1],X_elem[i][2],X_elem[i][3],modelCHG)

        padCHG1=pad_sequences(predCHG1.T, maxlen=padding_size,dtype='float32',padding='post',value=0.0)
        padCHG2=pad_sequences(predCHG2.T, maxlen=padding_size,dtype='float32',padding='post',value=0.0)
        padCHG3=pad_sequences(predCHG3.T, maxlen=padding_size,dtype='float32',padding='post',value=0.0)
        padCHG4=pad_sequences(predCHG4.T, maxlen=padding_size,dtype='float32',padding='post',value=0.0)

        X_C.append(np.concatenate((X_1[i].reshape(1,padding_size,360),padCHG1.T),axis=-1))
        X_H.append(np.concatenate((X_2[i].reshape(1,padding_size,360),padCHG2.T),axis=-1))
        X_N.append(np.concatenate((X_3[i].reshape(1,padding_size,360),padCHG3.T),axis=-1))
        X_O.append(np.concatenate((X_4[i].reshape(1,padding_size,360),padCHG4.T),axis=-1))
    X_C=np.vstack(X_C)
    X_H=np.vstack(X_H)
    X_N=np.vstack(X_N)
    X_O=np.vstack(X_O) 
    return X_C,X_H,X_N,X_O


