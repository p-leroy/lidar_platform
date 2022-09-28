# coding: utf-8
# Baptiste Feldmann

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal, stats
from sklearn.neighbors import NearestNeighbors

from ..tools import las


def findpeaks(line,treshold,size,nb_max):
    peaks,_=signal.find_peaks(line,treshold,width=size,rel_height=1)
    if len(peaks)>nb_max:
        peaks=peaks[0:nb_max]
        
    return peaks


def peaks_processing(point,line,params):
    peaks=findpeaks(line,*params)
    if len(peaks)>0:
        vlr=point.metadata['vlrs'][99+point['wavepacket_index']]
        point_loc=point['return_point_wave_location']
        anchor=point.XYZ+point_loc*[point['x_t'],point['y_t'],point['z_t']]

        peaks_time=(peaks+1)*vlr[3]
        peaks_coords=[anchor-(i*[point['x_t'],point['y_t'],point['z_t']]) for i in peaks_time]
        data_peaks=np.hstack([peaks_coords,np.reshape(line[peaks],(len(peaks),1)),np.reshape([len(peaks)]*len(peaks),(len(peaks),1)),
                              np.reshape(np.arange(1,len(peaks)+1),(len(peaks),1))])
        #names=['X','Y','Z','intensity','num_returns','return_num']
        return data_peaks


def viewerFWF(point,line,peaks=[]):
    wave_loc=point['return_point_wave_location']
    vlr=point.metadata['vlrs'][99+point['wavepacket_index']]
    x_vline=np.round(wave_loc/vlr[3])
    plt.figure()
    plt.plot(np.arange(0,len(line)*1000/vlr[3]),line,'k-')
    plt.plot([x_vline,x_vline],[min(line),max(line)],'k-')
    if len(peaks)>0:
        plt.plot(peaks*1000/vlr[3],line[peaks],"r+")
    
    plt.show()


def ringing_effect(data,tab_fwf,metadata,register=False):
    func=getattr(__import__("random"),"sample")
    analysis=[]
    names=metadata['col_names']
    window_minmax=(-5,25)
    seuil_intensite=3000
    if len(data[:,0])>50000:
        liste=func(range(0,len(data[:,0])),50000)
    else:
        liste=np.arange(0,len(data[:,0]))
    
    for i in liste:
        point=data[i,:]
        line=tab_fwf[i]
        wave_loc=point[names.index('return_point_wave_location')]
        vlr=metadata['vlrs'][99+point[names.index('wavepacket_index')]]
        x_vline=np.round(wave_loc/vlr[3])

        extrait_minmax=list(line[int(x_vline+window_minmax[0]):int(x_vline+window_minmax[1])])
        if np.max(extrait_minmax)<seuil_intensite:
            #Attention, delta T en picoseconds !!!
            delta_t=vlr[3]*(extrait_minmax.index(np.min(extrait_minmax))-extrait_minmax.index(np.max(extrait_minmax)))
            analysis+=[[np.min(extrait_minmax),np.max(extrait_minmax),
                        np.mean(line[int(x_vline+window_minmax[1])::]),
                        np.std(line[int(x_vline+window_minmax[1])::]),
                        delta_t]]
    analysis=np.array(analysis)
    if register:
        np.savez_compressed(register,["Min","Max","Mean_baselvl","Std_baselvl","delta_t"],analysis)

    select=func(range(0,len(analysis[:,0])),10000)
    Xdata=analysis[select,1]-analysis[select,2]
    Ydata=analysis[select,2]-analysis[select,0]
    slope,intercept,rvalue,pvalue,stderr=stats.linregress(Xdata,Ydata)
    return slope,intercept,rvalue


def __func(wave,pt):
    vlrs=pt.metadata['vlrs']
    idx_pt=np.round(pt['return_point_wave_location']/vlrs[int(99+pt['wavepacket_index'])][3],decimals=0)
    select=np.logical_and(wave>180,wave<215)
    base_lvl=np.median(wave[select])
    wave_clean=wave-base_lvl
    if len(wave_clean)<60:
        wave_clean=np.array(list(wave_clean)+[0]*(50-len(wave_clean)))
    elif len(wave_clean)>60:
        wave_clean=wave_clean[0:60]
    return wave_clean,idx_pt,base_lvl


def apply_shift(line,shift,value=0):
    if shift<0:
        new_line=list(line[abs(shift)::])+[value]*abs(shift)
    elif shift>0:
        new_line=[value]*shift+list(line[0:(-1*shift)])
    else:
        new_line=line
        
    return np.array(new_line)


def align_wave(lineA,lineB,peakA,peakB,baselvl=0):
    best=[0,0]
    factor=10

    if len(lineA)<len(lineB):
        lineB=lineB[0:len(lineA)]
    else:
        lineA=lineA[0:len(lineB)]
    diff=int(np.round(peakA-peakB,0))
    lineB=apply_shift(lineB,diff,baselvl)
    peakB+=diff

    lineA2=signal.resample(lineA,len(lineA)*factor)
    lineB2=signal.resample(lineB,len(lineB)*factor)

    for i in range(-20,20):
        score=stats.pearsonr(lineA2,apply_shift(lineB2,i,baselvl))
        if score[0]>best[0]:
            best=[score[0],i]

    lineB_aligned=signal.resample(apply_shift(lineB2,best[1],baselvl),len(lineB))
    return lineB_aligned,peakB+(best[1]/factor)


def compute_impulseResponse(data,wavefm,metadata,index_pt=-1,intensity_threshold=250,knn=10):
    f=getattr(getattr(__import__("sklearn"),"neighbors"),"NearestNeighbors")
    names=metadata['col_names']
    if index_pt==-1:
        idx_pt=np.random.randint(0,len(data[:,0]))
    else:
        idx_pt=int(index_pt)

    select=np.abs(data[:,names.index('intensity')]-data[idx_pt,names.index('intensity')])<intensity_threshold
    data_select=data[select,:]
    tab_select=list(np.array(wavefm)[select])

    neigh=f(n_neighbors=knn,n_jobs=40)
    neigh.fit(data_select[:,0:3])
    liste_select=neigh.kneighbors([data[idx_pt,0:3]],return_distance=False)[0]
    list_wavefm=list(np.array(tab_select)[liste_select])

    lineA,indexA,levelA=__func(tab_select[liste_select[0]],data_select[liste_select[0],:],metadata)
    totalLine=np.copy(lineA)
    baseLevel=[]
    for i in liste_select[1::]:
        line,index,level=__func(tab_select[i],data_select[i,:],metadata)
        baseLevel+=[level]
        totalLine=np.vstack([totalLine,align_wave(lineA,line)])

    return np.mean(totalLine,axis=0),np.std(totalLine,axis=0),baseLevel


def create_reference_waveform(data,wavefm,index_pt=-1,knn=10):
    if index_pt==-1:
        idx_pt=np.random.randint(0,len(data))
    else:
        idx_pt=int(index_pt)

    select =np.abs(data.intensity-data.intensity[idx_pt])<500
    dataSelect = las.filter_las(data, select)
    wavefmSelect = las.filter_wdp(wavefm, select)

    neigh=NearestNeighbors(n_neighbors=knn,n_jobs=40)
    neigh.fit(dataSelect.XYZ)
    listSelect=neigh.kneighbors([data.XYZ[idx_pt]],return_distance=False)[0]
    listWavefm=list(np.array(wavefmSelect)[listSelect])

    pt0= las.filter_las(dataSelect, listSelect[0])
    lineA,indexA,levelA=__func(wavefmSelect[listSelect[0]],pt0)
    lineA*=100/max(lineA[0:50])
    totalLine=np.copy(lineA)
    baseLevel=[levelA]
    peaks=[pt0.return_point_waveform_loc/1000]
    if pt0.metadata['vlrs'][99+pt0.wave_packet_desc_index][3]!=1000:
        raise ValueError("Temporal spacing doesn't equal to 1000 ns")

    for i in listSelect[1::]:
        ptX= las.filter_las(dataSelect, i)
        if ptX.metadata['vlrs'][99+ptX.wave_packet_desc_index][3]!=1000:
            raise ValueError("Temporal spacing doesn't equal to 1000 ns")

        line,index,level=__func(wavefmSelect[i],ptX)
        line*=100/max(line[0:50])
        baseLevel+=[level]
        result,peak2=align_wave(lineA,line,pt0.return_point_waveform_loc/1000,ptX.return_point_waveform_loc/1000)
        totalLine=np.vstack([totalLine,result])
        peaks+=[peak2]

    return np.mean(totalLine,axis=0),np.std(totalLine,axis=0),baseLevel,peaks

    
##def __compare_vlrs(liste):
##    numero=1
##    new_vlrs={}
##    first=True
##    for i in liste:
##        if first:
##            
##
##def merge_lasfwf(liste_filepath):
##    first=True
##    metadata_vlrs=[]
##    dictio={}
##    for i in liste_filepath:
##        f=las.readLAS(i,'fwf')
##        dictio[liste_filepath.index(i)]=f[0]
##        metadata_vlrs+=[f[1]['vlrs']]
##        if first:
##            names=f[1]['col_names']
##            first=False
##    __compare_vlrs(metadata_vlrs)
    
            

    

    
    
