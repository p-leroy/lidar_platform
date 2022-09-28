import requests
from bs4 import BeautifulSoup
import os

workspace=r'D:\Vertical_datum\calculs\IGN\test'+"//"

url="https://geodesie.ign.fr/contenu/fichiers/documentation/"

def download(url,outpath='',only_first_dir=False):
    chunk_size=100000
    url = url.replace(" ","%20")
    req = requests.get(url,stream=True)
    a = req.text
    soup = BeautifulSoup(a, 'html.parser')
    x = soup.findAll('a')

    for node in x:
        file_name=node.extract().get_text()
        file_name=file_name.replace(" ","%20")
        if(file_name[-1]=='/' and file_name[0]!='.'):
            if not only_first_dir:
                if len(outpath)>0:
                    os.mkdir(outpath+file_name)
                download(url+file_name,outpath+file_name)
                
        elif "." in file_name:
            print(url+file_name)
            if len(outpath)>0:
                data=requests.get(url+file_name,stream=True)
                with open(outpath+file_name,'wb') as f:
                    for chunk in data.iter_content(chunk_size):
                        f.write(chunk)

    
download(url,workspace)
