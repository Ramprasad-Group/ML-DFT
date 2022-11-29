import warnings
warnings.filterwarnings('ignore')
import numpy as np
import math
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import sys
sys.path.append(os.getcwd())
from inp_params import batch_size_fp, num_gamma, cut_off_rad, widest_gaussian, narrowest_gaussian
from operator import itemgetter
from joblib import dump, load
from sklearn import preprocessing
from sklearn.preprocessing import MaxAbsScaler

import time

import tensorflow as tf
import keras.backend as K
from keras.models import Sequential, Model
from keras.layers import Lambda,Input, concatenate,Dense, Conv1D,Dot,Flatten, Activation, Subtract, Reshape, Add, ELU, LeakyReLU, ReLU, TimeDistributed,PReLU,Dropout
from keras.preprocessing.sequence import pad_sequences

import pymatgen
from pymatgen import io
from pymatgen.io.vasp.outputs import Poscar
from pymatgen.io.vasp.outputs import Chgcar


from random import Random

import itertools
import h5py
import glob
import shutil

from keras.backend.tensorflow_backend import set_session
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH1 = os.path.join(ROOT_DIR, '../Trained_models/Scale_model_C.joblib')
CONFIG_PATH2 = os.path.join(ROOT_DIR, '../Trained_models/Scale_model_H.joblib')
CONFIG_PATH3 = os.path.join(ROOT_DIR, '../Trained_models/Scale_model_N.joblib')
CONFIG_PATH4 = os.path.join(ROOT_DIR, '../Trained_models/Scale_model_O.joblib')
class Input_parameters:
    cut_off_rad = cut_off_rad
    batch_size_fp = batch_size_fp
    widest_gaussian = widest_gaussian
    narrowest_gaussian = narrowest_gaussian
    num_gamma = num_gamma
    

inp_args=Input_parameters()


inp_args.list_sigma = np.logspace(math.log10(inp_args.narrowest_gaussian),math.log10(inp_args.widest_gaussian), num_gamma)
inp_args.list_gamma = 0.5 / inp_args.list_sigma ** 2
inp_args.num_gamma = num_gamma

sess = tf.compat.v1.Session()
init = tf.compat.v1.initialize_all_variables()

sess.run(init)
  

def fp_atom(poscar_data,supercell,elems_list):
    cart_grid=[]
    frac_grid=[]
    at_elem=[]
    sites_elem=[]
    num_elems = len(elems_list)
    element_fingerprint = []
    unique_cart_list=[]
    for elem in elems_list:
        elem_supercell = supercell.copy()
        list_remove = []
        for elem_remove in elems_list:
            if elem_remove != elem:
                list_remove.append(elem_remove)
        elem_supercell.remove_species(list_remove)
        at_elem.append(elem_supercell.num_sites)
        sites_elem.append(elem_supercell.sites.copy())
        list_neigh = elem_supercell.get_all_neighbors(float(inp_args.cut_off_rad),include_index=True)
        flat_list_neigh = list(itertools.chain.from_iterable(list_neigh))
        list_frac = [x[0].frac_coords for x in flat_list_neigh]
        extra_list=[x for x in elem_supercell.frac_coords]
        if list_frac:
            list_frac=np.concatenate((list_frac,extra_list), axis=0)
        else:
            list_frac=extra_list
        frac_array = np.array(list_frac)
        frac_array = np.ascontiguousarray(frac_array)
        cart_grid.append(np.array(elem_supercell.cart_coords))
        frac_grid.append(np.array(elem_supercell.frac_coords))

        if frac_array.shape[0]==0:
            frac_array=np.array(elem_supercell.frac_coords)
        unique_a = np.unique(frac_array.view([('', frac_array.dtype)]*frac_array.shape[1]))
        frac_array_unique = unique_a.view(frac_array.dtype).reshape((unique_a.shape[0], frac_array.shape[1]))
        unique_cart = np.dot(frac_array_unique, elem_supercell.lattice._matrix)
        unique_cart_list.append(unique_cart)
    cart_grid_K=np.concatenate(cart_grid, axis=0)
    frac_grid_K=np.concatenate(frac_grid, axis=0)
    padding_size=max(at_elem)

    def quad(cart_grid_K):
        quad_list=[]
        for num_e in range(len(unique_cart_list)):
            cart_atoms_K = K.variable(value=unique_cart_list[num_e])
            rad_diff = cart_grid_K - cart_atoms_K[:, None]
            rad = K.sqrt(rad_diff[:, :, 0] ** 2 + rad_diff[:, :, 1] ** 2 + rad_diff[:, :, 2] ** 2)
            rad_inv=1.0/K.sqrt(rad_diff[:, :, 0] ** 2 + rad_diff[:, :, 1] ** 2 + rad_diff[:, :, 2] ** 2)
            cut_off_rad = float(inp_args.cut_off_rad)

            del cart_atoms_K

            cut_off_func_K = (K.cos((K.minimum(rad, cut_off_rad) / cut_off_rad) * np.pi) + 1) / 2.0
            exp_term = []
            exp_term_vec=[]
            exp_term_ten=[]
            rad_list = []

            for nth_fp, gamma in enumerate(inp_args.list_gamma):
                norm = np.power(gamma / np.pi, 1.5)
                exp_term.append(norm * K.exp(-gamma * rad ** 2))
                tens1=norm * K.exp(-gamma * rad ** 2)
                tens2=rad**2
                p=tf.math.divide_no_nan(tens1, rad)
                h=tf.math.divide_no_nan(tens1, tens2)
                exp_term_vec.append(p)
                exp_term_ten.append(h)
                exp_term[nth_fp] = exp_term[nth_fp] * cut_off_func_K
                exp_term_vec[nth_fp]=exp_term_vec[nth_fp]*cut_off_func_K
                exp_term_ten[nth_fp]=exp_term_ten[nth_fp]*cut_off_func_K
                rad_list.append(K.sum(exp_term[nth_fp], axis=0))
#
            del cut_off_func_K
            rad_fp = K.stack(rad_list)

            rad_dir_list=[]
            for i in [0,1,2]:
                rad_list = []
                for nth_fp, gamma in enumerate(inp_args.list_gamma):
                    rad_list.append(K.sum((rad_diff[:,:,i] * exp_term_vec[nth_fp]), axis=0))

                rad_dir_list.append(K.stack(rad_list))
#
            dipole = K.sqrt(rad_dir_list[0]**2 + rad_dir_list[1]**2 + rad_dir_list[2]**2)
#
            rad_dir_list = []
#
            rad_1 = [0, 1, 2, 0, 1, 2]
            rad_2 = [0, 1, 2, 1, 2, 0]
#
            for i in [0, 1, 2, 3, 4, 5]:
                rad_list = []
                for nth_fp, gamma in enumerate(inp_args.list_gamma):
    
                    rad_list.append(K.sum(rad_diff[:,:,rad_1[i]] * rad_diff[:,:,rad_2[i]] * exp_term_ten[nth_fp], axis=0))
#
                rad_dir_list.append(K.stack(rad_list))
#
            del rad_list
            quad_1 = rad_dir_list[0] + rad_dir_list[1] + rad_dir_list[2]
#
            quad_2 = K.sqrt(K.abs(rad_dir_list[0] * rad_dir_list[1] + rad_dir_list[2] * rad_dir_list[0] + rad_dir_list[2] * rad_dir_list[1] - rad_dir_list[3] ** 2 - rad_dir_list[5] ** 2 - rad_dir_list[4] ** 2))
            quad_3 = K.pow(K.abs(rad_dir_list[0] * (rad_dir_list[1] * rad_dir_list[2] - rad_dir_list[4] ** 2) - rad_dir_list[3] * (
                        rad_dir_list[3]* rad_dir_list[2]- rad_dir_list[4] * rad_dir_list[5]) + rad_dir_list[5] * (
                        rad_dir_list[3] * rad_dir_list[4]- rad_dir_list[1] * rad_dir_list[5])), 1. / 3.)

            del rad_dir_list
            quad = K.concatenate([rad_fp, dipole, quad_1, quad_2, quad_3], 0)
            quad_list.append(K.transpose(quad))
#
        all_elem_quad=K.concatenate(quad_list,-1)

        return all_elem_quad
#
    def modelff():
        inp_grid = Input((3,))
        out=Lambda(quad)(inp_grid)
        modelff = Model(inputs=inp_grid, outputs=out)
        return modelff
    K.clear_session()
    modelff = modelff()
    num_atoms=cart_grid_K.shape[0]
    Y = modelff.predict(cart_grid_K, batch_size=inp_args.batch_size_fp,verbose=0)

    if num_elems==2:
        fp_el=Y.shape[1]
        zr=np.zeros((int(num_atoms),int(fp_el)))
        Y=np.concatenate((Y,zr),axis=-1)
        num_elems=4
        at_elem.append(0)
        at_elem.append(0)
    if num_elems==3:
        if elems_list[2]=='N':
            fp_el=Y.shape[1]/3
            zr=np.zeros((int(num_atoms),int(fp_el)))
            Y=np.concatenate((Y,zr),axis=-1)
            num_elems=4
            at_elem.append(0)
        else:
            fp_el=Y.shape[1]/3
            zr=np.zeros((int(num_atoms),int(fp_el)))
            Y_ch=Y[:,:int(2*fp_el)]
            Y_chn=np.concatenate((Y_ch,zr),axis=-1)
            Y_chno=np.concatenate((Y_chn,Y[:,int(2*fp_el):]),axis=-1)
            Y=Y_chno
            num_elems=4
            sites_elem.append(sites_elem[2])
            sites_elem[2]=0
            new_at_elem=[at_elem[0],at_elem[1],0,at_elem[2]]
            at_elem=new_at_elem
    print('at_elem', at_elem)
    num_atoms=cart_grid_K.shape[0]
    matrix_tot=[]
    matrixT_tot=[]
    cutoff_distance=5.0
    for pp in range(0,2):
        for x in sites_elem[pp]:
            pos=x.coords
            pos_frac=x.frac_coords
            neighs_list=list(itertools.chain.from_iterable(poscar_data.structure.get_neighbors(x,cutoff_distance)))
            group_lst=[neighs_list[i:i+4] for i in range(0, len(neighs_list), 4)]
            sorted_list=sorted(group_lst, key=itemgetter(1))
            v1=sorted_list[0][0].coords-pos
            v2=sorted_list[1][0].coords-pos
            u3=np.cross(v1,v2)
            u2=np.cross(v1,u3)
            u1=v1/np.linalg.norm(v1)
            u2=u2/np.linalg.norm(u2)
            u3=u3/np.linalg.norm(u3)
            matrx=np.transpose(np.array([u1,u2,u3]))
            matrix_tot.append(matrx)
    if at_elem[2] !=0:
        for x in sites_elem[2]:
            pos=x.coords
            pos_frac=x.frac_coords
            neighs_list=list(itertools.chain.from_iterable(poscar_data.structure.get_neighbors(x,cutoff_distance)))
            group_lst=[neighs_list[i:i+4] for i in range(0, len(neighs_list), 4)]
            sorted_list=sorted(group_lst, key=itemgetter(1))
            v1=sorted_list[0][0].coords-pos
            v2=sorted_list[1][0].coords-pos
            u3=np.cross(v1,v2)
            u2=np.cross(v1,u3)
            u1=v1/np.linalg.norm(v1)
            u2=u2/np.linalg.norm(u2)
            u3=u3/np.linalg.norm(u3)
            matrx=np.transpose(np.array([u1,u2,u3]))
            matrix_tot.append(matrx)
    if at_elem[3] !=0:
        for x in sites_elem[3]:
            pos=x.coords
            pos_frac=x.frac_coords
            neighs_list=list(itertools.chain.from_iterable(poscar_data.structure.get_neighbors(x,cutoff_distance)))
            group_lst=[neighs_list[i:i+4] for i in range(0, len(neighs_list), 4)]
            sorted_list=sorted(group_lst, key=itemgetter(1))
            v1=sorted_list[0][0].coords-pos
            v2=sorted_list[1][0].coords-pos
            u3=np.cross(v1,v2)
            u2=np.cross(v1,u3)
            u1=v1/np.linalg.norm(v1)
            u2=u2/np.linalg.norm(u2)
            u3=u3/np.linalg.norm(u3)
            matrx=np.transpose(np.array([u1,u2,u3]))
            matrix_tot.append(matrx)
    final_mat=np.vstack(np.array(matrix_tot))
    final_mat=np.reshape(final_mat,(num_atoms, 3, 3))

    return Y,final_mat,sites_elem,num_atoms,at_elem

def fp_chg_norm(Coef_at1,Coef_at2,Coef_at3,Coef_at4,X_3D1,X_3D2,X_3D3,X_3D4,padding_size):
    padCHG1=pad_sequences(Coef_at1.T, maxlen=padding_size,dtype='float32',padding='post',value=0.0)
    padCHG2=pad_sequences(Coef_at2.T, maxlen=padding_size,dtype='float32',padding='post',value=0.0)
    padCHG3=pad_sequences(Coef_at3.T, maxlen=padding_size,dtype='float32',padding='post',value=0.0)
    padCHG4=pad_sequences(Coef_at4.T, maxlen=padding_size,dtype='float32',padding='post',value=0.0)
    X_C=np.concatenate((X_3D1,padCHG1.T),axis=-1)
    X_H=np.concatenate((X_3D2,padCHG2.T),axis=-1)
    X_N=np.concatenate((X_3D3,padCHG3.T),axis=-1)
    X_O=np.concatenate((X_3D4,padCHG4.T),axis=-1)
    scalerC = load(CONFIG_PATH1)
    scalerH = load(CONFIG_PATH2)
    scalerN = load(CONFIG_PATH3)
    scalerO = load(CONFIG_PATH4)
    X_C=X_C.reshape(padding_size,X_C.shape[-1])
    X_H=X_H.reshape(padding_size,X_H.shape[-1])
    X_N=X_N.reshape(padding_size,X_N.shape[-1])
    X_O=X_O.reshape(padding_size,X_O.shape[-1])
    X_C = scalerC.transform(X_C)
    X_H = scalerH.transform(X_H)
    X_N = scalerN.transform(X_N)
    X_O = scalerO.transform(X_O)
    X_C=X_C.reshape(1,padding_size,X_C.shape[-1])
    X_H=X_H.reshape(1,padding_size,X_H.shape[-1])
    X_N=X_N.reshape(1,padding_size,X_N.shape[-1])
    X_O=X_O.reshape(1,padding_size,X_O.shape[-1])
    return(X_C,X_H,X_N,X_O)

def fp_norm(X_C,X_H,X_N,X_O,padding_size):
    a=X_C.shape[0]
    scalerC = load(CONFIG_PATH1)
    scalerH = load(CONFIG_PATH2)
    scalerN = load(CONFIG_PATH3)
    scalerO = load(CONFIG_PATH4)
    X_C=X_C.reshape(a*padding_size,X_C.shape[-1])
    X_H=X_H.reshape(a*padding_size,X_H.shape[-1])
    X_N=X_N.reshape(a*padding_size,X_N.shape[-1])
    X_O=X_O.reshape(a*padding_size,X_O.shape[-1])
    X_C = scalerC.transform(X_C)
    X_H = scalerH.transform(X_H)
    X_N = scalerN.transform(X_N)
    X_O = scalerO.transform(X_O)
    X_C=X_C.reshape(a,padding_size,X_C.shape[-1])
    X_H=X_H.reshape(a,padding_size,X_H.shape[-1])
    X_N=X_N.reshape(a,padding_size,X_N.shape[-1])
    X_O=X_O.reshape(a,padding_size,X_O.shape[-1])
    return(X_C,X_H,X_N,X_O)
