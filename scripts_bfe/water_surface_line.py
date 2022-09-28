import numpy as np
from shapely import wkt
import shapely
import shapely.ops
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
#import shapefile as sf
import pycrs
import itertools
import scipy.spatial as scp


##def readSHP(filepath):
##    file=sf.Reader(filepath)
##    data=file.shapes()
##    listGeom=[shapely.geometry.LineString(i.points) for i in data]
##    return listGeom

class Alphashape(object):
    def __init__(self,points2D,alpha=100):
        self.points=points2D
        self.alpha=alpha
        self.compute_alphashape()
        
    def compute_alphashape(self):
        # More you decrease alpha factor, greater the constraint will be on alphashape
        # More you increase alpha factor, more the alphashape will be a convex hull
        print("[Alphashape] compute alphashape...",end=' ')
        tri=scp.Delaunay(self.points)
        edges=set()
        edge_points=[]
        for idx in tri.vertices:
            triangle=self.points[idx,:]
            length=[np.linalg.norm(triangle[i%3]-triangle[(i+1)%3]) for i in [0,1,2]]
            s=np.sum(length)*0.5
            area=s*np.prod(s-length)
            if area>0:
                area=np.sqrt(area)

            if np.prod(length)/(4*area) < self.alpha:
                for i,j in itertools.combinations(idx,r=2):
                    if (i,j) not in edges and (j,i) not in edges:
                        edges.add((i,j))
                        edge_points.append(self.points[[i,j],:])
        m=shapely.geometry.MultiLineString(edge_points)
        triangles=list(shapely.ops.polygonize(m))
        self.alphashape=shapely.ops.cascaded_union(triangles)
        print("done !")

    def print_alphashape(self):
        if type(self.alphashape)==shapely.geometry.polygon.Polygon:
            a=self.alphashape.exterior.xy
            plt.fill(a[0],a[1],alpha=2,edgecolor="red",facecolor="blue")
        else:
            for i in self.alphashape:
                a=i.exterior.xy
                plt.fill(a[0],a[1],alpha=2,edgecolor="red",facecolor="blue")
        plt.show()

    def filter_voronoi(self):
        # step 1
        print("[Alphashape] filtering Voronoi diagram...",end=' ')
        coords=np.stack([*self.alphashape.exterior.xy]).transpose()
        vor=scp.Voronoi(coords)
        edge_points=[]

        for idx in vor.ridge_vertices:
            line=shapely.geometry.LineString(vor.vertices[idx,:])
            if self.alphashape.contains(line):
                edge_points.append(vor.vertices[idx,:])
        geom=shapely.geometry.MultiLineString(edge_points)
        self.lines=shapely.ops.linemerge(geom)
        print("done !")

    def filter_lines(self,beginNum=2,endNum=3):
        # step 2
        print("[Alphashape] filtering geometry..",end=' ')
        for num in range(beginNum,endNum):
            print(num,end=' ')
            listVertices=[np.round(np.array(i.xy).transpose()[[0,-1],:],decimals=3) for i in self.lines]
            listVerticesAll=np.reshape(listVertices,(len(listVertices)*2,2))
            listAllPoints=[np.array(i.xy).transpose() for i in self.lines]
            listNum=[len(np.array(i.xy).transpose()[:,0]) for i in self.lines]
            edge_points=[]

            for i in range(0,len(listNum)):
                pts=listVertices[i]
                test=[np.logical_and(len(np.where(pts[0,0]==listVerticesAll[:,0])[0])>1,len(np.where(pts[0,1]==listVerticesAll[:,1])[0])>1),
                      np.logical_and(len(np.where(pts[1,0]==listVerticesAll[:,0])[0])>1,len(np.where(pts[1,1]==listVerticesAll[:,1])[0])>1)]
                if any(test) == all(test) or listNum[i] > num :
                    edge_points.append(listAllPoints[i])
            self.lines=shapely.ops.linemerge(shapely.geometry.MultiLineString(edge_points))
        print('done !')

    def save_geometry(self,filepath,epsg=2154):
        # Filepath do not contain extension at the end
        w=sf.Writer(filepath)
        w.field('name','C')
        compt=1
        for i in self.lines:
            coords=np.array(i.xy).transpose()
            w.line([coords])
            w.record('segment'+str(compt))
            compt+=1
        w.close()
        f=open(filepath+".prj",mode='w')
        f.write(pycrs.parse.from_epsg_code(epsg).to_ogc_wkt())
        f.close()
    

class WaterLine(object):
    def __init__(self,rawlines,model,surface_Z):
        self.rawlines=rawlines
        self.model_knn=model
        self.surf_Z=surface_Z
        self.data={}
        self.metadata={"connection":{},"downstream":{}}
        self.computeData()
        self.findConnect()
        self.distDownstream()

    def __func(self,pt1,pt2):
        step=2
        vect=pt2-pt1
        dist=np.linalg.norm(vect)
        #print(vect,dist,sep='\t')
        vect_norm=vect/dist
        liste=np.arange(0,int(dist),step)
        listPts=[]
        for i in liste:
            listPts+=[pt1+vect_norm*i]
        return listPts
        
    def computeData(self):
        for i in range(0,len(self.rawlines)):
            coords=self.computeXYZ(i)
            XY=coords[:,0:2]
            dist=np.linalg.norm(XY[1::,:]-XY[0:-1,:],axis=1)
            dist=np.reshape(np.append(dist,0),(len(coords[:,0]),1))
            self.data[i]=np.append(coords,dist,axis=1)
            pts_bas=coords[[0,-1],2]
            if pts_bas[0]>pts_bas[1]:
                aval="last"
            else:
                aval="first"
            self.metadata["downstream"][i]=aval
            
    def computeXYZ(self,index):
        test=[]
        rawline=self.rawlines[index].xy
        for i in range(0,len(rawline[0])-1):
            test+=self.__func(np.array([rawline[0][i],rawline[1][i]]),np.array([rawline[0][i+1],rawline[1][i+1]]))
        test+=[[rawline[0][-1],rawline[1][-1]]]
        test=np.array(test)
        listIndex=self.model_knn.kneighbors(test,return_distance=False)
        listAlti=np.reshape(np.round([np.median(self.surf_Z[i]) for i in listIndex],2),(len(test[:,0]),1))
        return np.append(test,listAlti,axis=1)

    def findConnect(self):
        models=[NearestNeighbors(n_neighbors=1,n_jobs=50).fit(self.data[i][:,0:2]) for i in self.data.keys()]
        connection={}
        prec=0.1
        for i in self.data.keys():
            connection[i]={'first':[],'last':[]}
            extremity=self.data[i][[0,-1],0:2]
            numLine=[None,None]
            idxLine=[None,None]
            found=[False,False]
            for c in self.data.keys():
                if i!=c and (not found[0] or not found[1]):
                    dist,index=models[c].kneighbors(extremity)
                    for m in [0,1]:
                        if dist[m]<prec:
                            numLine[m]=c
                            idxLine[m]=index[m][0]
                            found[m]=True
            connection[i]['first']=[numLine[0],idxLine[0]]
            connection[i]['last']=[numLine[1],idxLine[1]]
        self.metadata['connection']=connection

    def cumDist(self,index):
        lenSeg=self.data[index][:,3]
        if self.metadata['downstream'][index]=="first":
            dist=[0]+[np.sum(lenSeg[0:i]) for i in range(1,len(lenSeg))]
        else:
            dist=[np.sum(lenSeg[i::]) for i in range(0,len(lenSeg))]
        return dist
            
    def distDownstream(self):
        for i in self.data.keys():
            cumulativeDistance=self.cumDist(i)
            self.data[i]=np.append(self.data[i],np.reshape(cumulativeDistance,(len(cumulativeDistance),1)),axis=1)

        for i in self.data.keys():
            connect_down=self.metadata['connection'][i][self.metadata['downstream'][i]]
            if connect_down[0] is not None:
                self.data[i][:,4]+=self.data[connect_down[0]][connect_down[1],4]

    def save_data(self,filepath):
        temp=[]
        for i in self.data.keys():
            a=np.append(self.data[i],np.reshape([i]*len(self.data[i][:,0]),(len(self.data[i][:,0]),1)),axis=1)
            temp+=[a]

        #temp=[test.data[i] for i in test.data.keys()]
        merge=np.concatenate(temp,axis=0)
        np.savetxt(filepath,merge,fmt='%.3f',delimiter=';',header='X;Y;Z;lenSeg;cumul;index')
            
                   
if __name__=='__main__':
    workspace=r'G:\RENNES1\Loire_totale_automne2019\Loire_Briare-Sully-sur-Loire\05-Traitements'+'//'
    inFile="water_surface.csv"
    surface_water=r'G:\RENNES1\Loire_totale_automne2019\Loire_Briare-Sully-sur-Loire\05-Traitements\C2_ground_thin_1m_watersurface_smooth5.laz'
    data= tools.lastools.read(surface_water)

##    # Compute alphashape
##    AS=Alphashape(data.XYZ[:,0:2],50)
##    AS.print_alphashape()
##    AS.filter_voronoi()
##    AS.filter_lines(2,8)
##    geometry=copy.deepcopy(AS.lines)
##    AS.save_geometry(workspace+outFileRoot+"_rawlines")
##    del AS

    #geometry=readSHP(workspace+"tempfile.shp")
    f=open(workspace+inFile)
    geometry=[]
    for line in f.readlines()[1::]:
        geometry+=[wkt.loads('LINESTRING '+line[18:-3])]    
    
    # Compute 3D lines
    tree=NearestNeighbors(n_neighbors=10,n_jobs=40)
    tree.fit(data.XYZ[:,0:2])
    Line3D=WaterLine(geometry,tree,data.XYZ[:,2])
    Line3D.save_data(workspace+inFile[0:-4]+"_final.csv")







##test=Line(rawlines,tree,data[:,2])
##
##temp=[]
##for i in test.data.keys():
##    a=np.append(test.data[i],np.reshape([i]*len(test.data[i][:,0]),(len(test.data[i][:,0]),1)),axis=1)
##    temp+=[a]
##
###temp=[test.data[i] for i in test.data.keys()]
##merge=np.concatenate(temp,axis=0)
##np.savetxt(workspace+"test_output4.csv",merge,fmt='%.3f',delimiter=';',header='X;Y;Z;lenSeg;cumul;index')





