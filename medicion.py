import numpy as np
from configparser import ConfigParser
import cv2
import os
import matplotlib.pyplot as plt
from numpy.core.arrayprint import dtype_is_implied
from skimage import morphology

import sys# las sisguientes dos lineas ayudan a ubicar al archivo calibracion.py
sys.path.append("/home/estufab4/Desktop/flujoBagazo/codigo/calibracion/")
from calibracion import puntos
# segementador de objetos con alto contraste
#la función puede semegtar un objeto blanco en un fondo negro, negro= False
#la funcioón puede segmentar un objeto negro en un fondo blanco, negro=True
# def segmetador(img,negro=True):
#     img_gray=cv2.cvtColor(img.copy(), cv2.COLOR_RGB2GRAY)
#     #img_= cv2.medianBlur(img_gray,15)
#     if negro:
#         criterio= cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU
#     else:
#         criterio=cv2.THRESH_BINARY+cv2.THRESH_OTSU
#     thr,mascara =cv2.threshold(img_gray,0,1,criterio)
#     mascara.dtype=bool## remove_small solo funciona con variables binarias
#     mascara=morphology.remove_small_objects(mascara,50,connectivity=1)
#     mascara.dtype=np.uint8
#     img_mascara=cv2.bitwise_and(img,img,mask=mascara)
#     return thr,mascara,img_mascara



#La función permitira graficar datos de dos dimensiones
#La función podrá graficar todos los datos en un subplot o en un solo plot
#La función tendra auto ajuste en el rango de los ejes.
#list_x: datos eje horizontal
#list_y: datos eje vertical
#filas: numero de filas del subplot
#colum: numero de columnas del subplot
#list_titulos: argumento opcional para colocar el titulo cada grafica, debe tener las misma dimension que la listas de datos.
# def plot(list_x,list_y,filas=1,colum=1,list_titulos=[],ejes=[],tipo=0):
#     plt.figure(figsize=(10, 10), facecolor='w', edgecolor='b')
    
#     if tipo:
#         colors = "bgrcmykw"
#         for i in range(len(list_x)):
#             plt.plot(list_x[i],list_y[i],c=colors[i],linewidth=1)
#             #lt.yscale('linear', linthreshy=10)
#             plt.axis([np.amin(list_x[i]), np.max(list_x[i]), np.amin(list_y[i]), np.max(list_y[i])])
#             plt.grid(color='k', alpha=1, linestyle='dashed', linewidth=1)
#             plt.xlabel(ejes[0][0], fontsize=14, color='k')
#             plt.ylabel(ejes[0][1], fontsize=14, color='k')
#             plt.axis('auto')
#             plt.grid(color='k', alpha=1, linestyle='dashed', linewidth=1)
#             plt.tick_params(labelsize=12)
#             plt.grid(True)
#             if len(list_titulos):
#                 plt.title(list_titulos[0],fontsize=40)
#     else:
#         if len(list_x)!=len(list_y):
#             print("numero incorrect de datos")
#         else:
#             numero=len(list_x)
#             titulos=len(list_titulos)
#             if titulos!=numero:
#                 titulos=0
#                 print("numero de titulos incorrecto")
          
#             for i in range(numero):
#                 plt.subplot(filas,colum,i+1)
#                 plt.scatter(list_x[i],list_y[i],linewidth=1)
#                 #lt.yscale('linear', linthreshy=10)
#                 #plt.axis([np.amin(list_x[i]), np.max(list_x[i]), np.amin(list_y[i]), np.max(list_y[i])])
#                 plt.axis('auto')
#                 #plt.plot(list_x[i],list_y[i],linewidth=10.0)
#                 plt.xlabel(ejes[i][0], fontsize=14, color='k')
#                 plt.ylabel(ejes[i][1], fontsize=14, color='k')
#                 plt.grid(color='k', alpha=1, linestyle='dashed', linewidth=1)
#                 plt.tick_params(labelsize=12)
#                 plt.grid(True)
#                 if titulos>0 : 
#                     plt.title(list_titulos[i],fontsize=40)
            

#     plt.show()




# la clase carga los datos que se obtiene de la calibración.
#path direccion absoluta del lugar donde se encuentra el archivo de configuración
#nombre del archivo de configuración
# la clase carga los datos que se obtiene de la calibración.
#path direccion absoluta del lugar donde se encuentra el archivo de configuración
#nombre del archivo de configuración
import numpy as np
from configparser import ConfigParser
import sys# las sisguientes dos lineas ayudan a ubicar al archivo calibracion.py
sys.path.append("/home/estufab4/Desktop/flujo de bagazo/codigo/calibracion/")
from calibracion import *
# la clase carga los datos que se obtiene de la calibración.
#path direccion absoluta del lugar donde se encuentra el archivo de configuración
#nombre del archivo de configuración
class configuracion:
    def __init__(self,path,nombre_config,area=False,pendientes2=False):
        self.path=path
    #secciones
        self.secciones=['homografia', 'calibracion_uv_depth', 'calibracion_uv_rgb','calibracion_z']
        #caracteristicas rgb
        self.rgb_mtx=0
        self.rgb_intrinseca=0
        self.rgb_coef_dist_=0
        self.rgb_roi=0
        self.rgb_inversa=0
        #caracteristicas profundidad
        self.depth_mtx=0
        self.depth_intrinseca=0
        self.dpth_coef_dist_=0
        self.depth_roi=0
        self.depht_inversa=0
        #cambio de dominio
        self.homografia=0
        #calibracion z
        # if pendientes2, self.pendientes=0
        self.pendientes=0
        self.interceptos=0
        #ejecución de funciones
        #cargar archivo de configuracion
        self.config = ConfigParser()
        #revisar existencia del archivo  y comprobar existencia de todas las secciones
        ret1,ret2=self.comprobar(nombre_config)
        if not(ret1):
            print("el achivo no existe ")
        if not ret2:
            print("hay un numeor invalido de secciones")
        #cargar valores
        self.rgb_mtx,self.rgb_intrinseca,self.rgb_inversa,self.rgb_coef_dist,self.rgb_roi=self.calibracion_uv("rgb")
        self.depth_mtx,self.depth_intrinseca,self.depth_inversa,self.depth_coef_dist,self.depth_roi,=self.calibracion_uv("depth")
        self.homografia=self.calibracion_homografia()
        #print(pendientes2)
        if pendientes2:
            self.pendientes2,self.pendientes,self.interceptos=self.calibracion_z2()
        else:
            self.pendientes,self.interceptos=self.calibracion_z()
        if area: # debido a compatibilidad
            self.polinomio=self.calibracion_area()
        else:
            self.polinomio=0
            
    # comprobar si todas las secciones se encuentran
    def comprobar(self,nombre):
     #revisar existencia del archivo
        ret1=self.config.read('{}{}.ini'.format(self.path,nombre))
    # comprobar que se encuentren todas las secciones
        secciones=self.config.sections()
        ret2=True
        for i in range(len(self.secciones)):
            ret2=ret2 and(self.secciones[i]in secciones)
        return ret1, ret2
    def calibracion_uv(self,tipo):
        #hay que convertir el texto a matriz 
        datos=(self.config["calibracion_uv_{}".format(tipo)])
        mtx=self.preparar_dato(datos["mtx"],3,3)
        mt_intrinseca=self.preparar_dato(datos["mt_intrinseca"],3,3)
        mt_inversa=self.preparar_dato(datos["mt_inversa"],3,3)
        roi=self.preparar_dato(datos["roi"],1,4)
        coef_dist=self.preparar_dato(datos["coef_dist"],1,5)
        return mtx,mt_intrinseca,mt_inversa,coef_dist,roi
    def calibracion_homografia(self):
        datos=self.config["homografia"]
        homografia=self.preparar_dato(datos["homografia"],3,3)
        return homografia
    def calibracion_z(self):
        datos=self.config["calibracion_z"]
        shape=(self.preparar_dato(datos["shape"],1,2).reshape(-1)).astype(np.int)
        margen_v=int(datos["margen_v"])
        margen_h=int(datos["margen_h"])
    
        pendientes_marco=[]
        interceptos_marco=[]
        alto=shape[0]-2*margen_v
        ancho=shape[1]-2*margen_h
        for i in range(alto*ancho):
            pendientes_marco.append(float(datos["pendiente{}".format(i)]))
            interceptos_marco.append(float(datos["intercepto{}".format(i)]))
        pendientes_marco=np.array(pendientes_marco)
        interceptos_marco=np.array(interceptos_marco)
        # colocar mapa en dimensiones de la imagen
        pendientes=np.ones(shape)
        interceptos=np.zeros(shape)
        pendientes[margen_v:-margen_v,margen_h:-margen_h]=pendientes_marco.reshape(alto,ancho)
        interceptos[margen_v:-margen_v,margen_h:-margen_h]=interceptos_marco.reshape(alto,ancho)
        return pendientes,interceptos
    def calibracion_z2(self):
        datos=self.config["calibracion_z"]
        shape=(self.preparar_dato(datos["shape"],1,2).reshape(-1)).astype(np.int)
        pendientes2=[]
        pendientes=[]
        interceptos=[]

        for i in range(shape[0]*shape[1]):
            pendientes2.append(float(datos["pendiente2_{}".format(i)]))
            pendientes.append(float(datos["pendiente{}".format(i)]))
            interceptos.append(float(datos["intercepto{}".format(i)]))
        pendientes2=np.array(pendientes2)
        pendientes=np.array(pendientes)
        interceptos=np.array(interceptos)
        # colocar mapa en dimensiones de la imagen
        pendientes2=pendientes2.reshape(shape[0],shape[1])
        pendientes=pendientes.reshape(shape[0],shape[1])
        interceptos=interceptos.reshape(shape[0],shape[1])
        return pendientes2,pendientes,interceptos
    def calibracion_area(self):
        datos=self.config["area"]
        polinomio=self.preparar_dato(datos["polinomio"],-1,1).reshape(-1)
        return polinomio
    #dato:string que contine un vector fila en formato texto. el string tambien contine las llaves que indican que es un vector fila
    # a pesar de ser un estring, cada elemento se encuentra separado por un espacio.
    #nf:para convertir la entrada a una matriz. núero de filas
    #nc:para converitr la entrada a una matriz. número de columnas
    #nf
    def preparar_dato(self,dato,nf,nc):
        dato=dato.lstrip("[")#quitar llave lado izquierdo
        dato=dato.rstrip("]")#quitar corchete lado derecho
        dato=dato.split()# separa el string teniendo en cuenta los espacios, regresa  una lista de estring.
        dato=np.array(dato,dtype=np.float).reshape(nf,nc)#convirte la lista a un array numpy. cada elemento se convierte a float64. luego se convierte en una matriz
        return dato
        
        
        

#el contructor de la clase resive el nombre y direccion absolta del lugar donde se encuentra el archivo de configuracion que contiene los parametros de calibración
# esta clase tiene funciones que resiven una imagen y le quitan la distorcion ya sea para el sensor rgb o el sensor de profundidad
#esta clase tambien tiene una función que lleva una imagen del dominio rgb al dominio de profundidad
# la clase posee una funcion para poder abrir un par de imagenes que reprensenten la misma escena
class imagen(configuracion):
    def __init__(self,path,nombre_config,pendientes2=False):
        
        configuracion.__init__(self,path,nombre_config,pendientes2=pendientes2)
        self.zmin=np.array(4000,dtype=np.float64)
        self.zmax=np.array(22000,dtype=np.float64)

    # quitar distorsion causado por lentes a la imagen rgb
    def distorsion_rgb(self,img):

        dst = cv2.undistort(img, self.rgb_mtx, self.rgb_coef_dist, None, self.rgb_intrinseca)

        return dst
    # quitar distorsion causado por lentes a la imagen de profundidad
    def distorsion_depth(self,img):
     
        dst = cv2.undistort(img.copy(), self.depth_mtx, self.depth_coef_dist, None, self.depth_intrinseca)
        
        return dst
    #quitar distorsión causada por lentes a par de imagenes
    def distorsion(self,img_rgb,img_depth):
        dst_rgb = cv2.undistort(img_rgb, self.rgb_mtx, self.rgb_coef_dist, None, self.rgb_intrinseca)
        dst_depth = cv2.undistort(img_depth, self.depth_mtx, self.depth_coef_dist, None, self.depth_intrinseca)
        return dst_rgb,dst_depth.astype(float)
    def distorsion_z(self,img_depth,filtrar):
        dst_depth=img_depth.copy()
        if filtrar:
            dst_depth=self.filtrar_depth(dst_depth,blur=3,radio=40,sigma_espacio=80,sigma_color=50)
        dst_depth=dst_depth*self.pendientes+self.interceptos
        return dst_depth.astype(np.float32)
    def normalizardst(self,img):
        return (img-self.zmin)/(self.zmax-self.zmin)
    def recuperar(self,img):
        mask=img.copy()
        mask[mask<3000]=1
        mask[mask!=1]=0
        mask=mask.astype(np.uint8)
        dst = cv2.inpaint(img.copy(),mask,1,cv2.INPAINT_TELEA)
        
        return dst.astype(float)
        
    def denormalizardst(self, img):
        return img*(self.zmax-self.zmin)+self.zmin
    def distorsion_z2(self,img_depth,filtrar):
        dst_depth=np.array(img_depth,dtype=np.float64)

        if filtrar:
            dst_depth=self.filtrar_depth(dst_depth,blur=3,radio=40,sigma_espacio=80,sigma_color=50)
        dst_depthNorm=self.normalizardst(dst_depth)
        dst_depthNorm2=np.power(dst_depthNorm.copy(),2)
        dst_depth=dst_depthNorm2*self.pendientes2+dst_depthNorm*self.pendientes+self.interceptos
        dst_depth=self.denormalizardst(dst_depth)
        #print(dst_depth.mean())
        
        return dst_depth
    # pasar una imagen del dominio rgb al dominio de profundidad
    def rgb2_depth(self,img):
        img_depth= cv2.warpPerspective(img.copy(), self.homografia, (img.shape[1], img.shape[0]))#primero ancho y luego alto
        return img_depth
    def filtrar_depth(self,img_depth,blur,radio,sigma_espacio,sigma_color):
        img=img_depth.copy().astype(np.float64)
        #img[img==0]=img.mean()
        img=cv2.blur(img,(blur,blur))
        img = cv2.bilateralFilter(img,radio,sigma_espacio, sigma_color)
        return img
    
    # abre el par de imagenes rgb y profundidad que esten captando la misma escena
    #los nombre desben estar en el formato indicador_nombre. indicardor="rgb" para la imagen del sensor rgb
    #indicador="depth" para la imagen del sensor de profundidad. 
    #nobre debera ser el mismo para el par de imagenes que representen la misma escena.
    def abrir_par_img(self,path,indicador_rgb,indicador_depth,nombre):
        path_rgb=os.path.join(path, '{}_{}.png'.format(indicador_rgb,nombre))
        path_depth=os.path.join(path, '{}_{}.png'.format(indicador_depth,nombre))
        #confirmar existencia de ambas imagenes
        ret_2=cv2.haveImageReader(path_depth)
        ret_1=cv2.haveImageReader(path_rgb)
        img_rgb=0;img_depth=0
        ret=ret_1 and ret_2#
        if ret:
            img_depth=cv2.imread(path_depth,cv2.IMREAD_UNCHANGED)[:,:,0]
            img_rgb=cv2.imread(path_rgb)
           
        return ret,img_rgb,img_depth
    # aplicación de las funciones declaras en esta clase para poder realizar mediciones
    def preprocesar_imgs(self,img_rgb,img_depth,f_segmentador):
        #quitar distorsion z
        img_depth=self.distorsion_z(img_depth,filtrar=True)
     
        #quitar distorsion uv
        img_rgb,img_depth=self.distorsion(img_rgb,img_depth)
        # cambiar de dominio rgb a depth
        img2_depth=self.rgb2_depth(img_rgb)
        # encontrar mascara
        mascara=f_segmentador(img2_depth)
        return img_depth,mascara
    
    
# la clase podra calcular la distancia entre los puntos medio de los circulos por fila del patron.
#la  tendra una función que resive la informacion de una escena. 2 imagenes. imagen rgb e imagen de profundidad que represente la misma escena
#a la imagen rgb se le quita la distorsion se la lleva al dominio de profundidad y alli se encuentran los puntos uv, puntos imange.
# se calcula los puntos (x,y,z) a travez de los puntos uv y la informacion de la camara de profundidad.
# se encuentra la distancia entre los puntos (X,Y,Z) y se calcula el error con respecto a las medidas del patron
# en el constructor se inicializa las variables de la calibración y la iformacion util para extraer los puntos.

class longitud(imagen,puntos):
    # path_configuracion: dirección absoluta del lugar donde se encuentra el archivo de configuración
    #nombre_configuración: nombre del archivo configuración sin la extensión .ini
    #d:diametro de circulos en milimetros
    # nf: numero de filas del patron
    #nc: numero de circulos por fila del patron
    def __init__(self,path_configuracion,nombre_configuracion,d,nf,nc):
        imagen.__init__(self,path_configuracion,nombre_configuracion)
        puntos.__init__(self,nf,nc)
        # variables de isntancia
        self.distancia_patron=self.formar_distancia(d)# distancia original entre puntos contiguos del patron. tiene en cuenta que los puntos estan agrupados por fila, los puntos de cada fila estan ordenados de izquierda a derecha, y las filas de arriba hacia abajo.
        self.error_promedio=0
    # abre el par de imagenes rgb y profundidad que esten captando la misma escena
    #los nombre desben estar en el formato indicador_nombre. indicardor="rgb" para la imagen del sensor rgb
    #indicador="depth" para la imagen del sensor de profundidad. 
    #nobre debera ser el mismo para el par de imagenes que representen la misma escena.
    #pasa la imagen rgb al dominio de profundidad
    #
    def obtener_uvz(self,path, indicador1,indicador2,nombre):
        #abrir imagenes
        ret,img_rgb,img_depth=self.abrir_par_img(path,indicador1,indicador2,nombre)
        #quitar distorision z
        img_depth=self.distorsion_z(img_depth,filtrar=True)
        #quitar distorsion
        img_rgb,img_depth=self.distorsion(img_rgb,img_depth)
        
        # cambiar de dominio
        img2_depth=self.rgb2_depth(img_rgb)
        # hallar puntos medios entre los circulos de una misma fila
        ret,_,puntos_uv=self.extraer(img2_depth,0)# es una imagen en el dominio de profundidad pero su caracteristicas siguen siendo de color. no necesita mejora
        # informar si hubo detección de puntos
        print("hubo de detección de puntos?: {}".format(ret))
        #obtener cordenada z # numpy indexa fila x columna, opencv indexa ancho por lato. por lo tanto hay que invertir los indices.
        if ret:
            z=[]
            for punto in puntos_uv:
                fila=int(np.around(punto[0][1]))
                col=int(np.around(punto[0][0]))
                valor=img_depth[fila,col]

                z.append(valor)
            z=np.array([z])
        #dibujar puntos medios sobre la imagen de profundidad
        img_puntos=self.dibujar(img_depth,puntos_uv,1)
        return ret,puntos_uv,z,img_puntos
    # la función resive puntos uv y los trasforma a puntos X,Y,Z
    # reproyecta los puntos uv encontrados en la imagen y los lleva al espacio.las cordenadas son cartesians y su origen  es el foco de la camara
    def obtener_xyz(self,puntos_uv,z):
        
        # los puntos estan almacenados en vectores fila [u,v]. para la multiplicacion matricial se los lleva a vectores tipo columna [u;v;1]
        puntos_uv=puntos_uv.reshape(-1,2) # desepaquetar. de shape(n,1,2) hacia shape(n,2)
        puntos_u=puntos_uv[:,0]# extraer todas las componentes u de los puntos
        puntos_v=puntos_uv[:,1]# #extraer todas las componentes v
        unos=np.ones(puntos_uv.shape[0])
        puntos_uv1=np.array((puntos_u,puntos_v,unos))# formar vectores columna (u;v;1)
        puntos_xy1_n=np.dot(self.depth_inversa,puntos_uv1)# calcular cordenadas normalizadas
        puntos_xyz=z*puntos_xy1_n# calular cordenadas espaciales de la forma [X;Y;Z]. son vectores columna
        #devolver puntos a forma vector fila [(x,y,z)].separando los putos
        puntos_xyz=np.concatenate((puntos_xyz[0].reshape(-1,1),puntos_xyz[1].reshape(-1,1),puntos_xyz[2].reshape(-1,1)),axis=1)
        return (puntos_xyz) #entregar array de la forma shape(n,3)
    #encuentra la distancia euclidiana entre los puntos contiguos XYZ y los compara  con la distancia de construcción de los puntos del patron
    def calcular_distancia(self,puntos_xyz):
        distancias=[]
        error_distancias=[]
        for i in range(puntos_xyz.shape[0]-1):
            vector=puntos_xyz[i+1]-puntos_xyz[i]
            distancia=np.sqrt(np.sum(np.power(vector,2)))
            distancias.append(distancia)
        distancias=np.array(distancias).reshape(-1,1)/10
        error_distancias=np.absolute(self.distancia_patron-distancias)
        error_distancias=100*error_distancias/self.distancia_patron
        self.error_promedio=np.sum(error_distancias)/error_distancias.size
        return distancias,error_distancias,self.error_promedio
    #la distancia se calcula entre dos puntos contiguos
    #d=diametro de los circulos
    #nf=numero de filas del patron
    #nc=numero de circulos por fila
    #los puntos estan agrupados por filas, los puntos de cada fila se encuentran ordenados de izquierda a derecha y las filas de arriba hacia abajo
    def formar_distancia(self,d):
        nc=self.nc-1# al ser puntos medio entre circulos se pierde un punto por cada fila
        distancias_patron=np.ones((self.nf*nc-1,1))#al ser distancia sobre puntos contiguos la informacion disminuye en un dato
        #formar distancias teniendo en cuenta la estructura del patron
        for i in range(distancias_patron.size):
            if (i+1)%nc:# puntos contiguos de una misma fila
                distancias_patron[i]=2*d
            else:#puntos contiguos, punto extremo derecho fila superior, punto extremo izquierdo fila inferior
                if (i+1)%10:# por se patron asimetrico,las distancias entre extremos cambian de una fila a otra
                    distancias_patron[i]=np.sqrt((2*(nc-1)*d-d)**2+d**2)#fila impar
                else:
                    distancias_patron[i]=np.sqrt((2*(nc-1)*d+d)**2+d**2)#fila par
        return distancias_patron
###########################################################################################################
#la clase debe cargar el archivo de configuracin que contiene los datos de la calibración
#la case debe funcionar independiente del segmentador
#la clase tendra una función que resive un par de imagenes. RGB y profundidad y le quita la distorsión tanto en uv , como en z. ademas realiza un filtro promedio y bilateral en z.
#la clase tendra una función que resiva una imagen de profundidad y una mascara y apartir de ello calculara area.
#la clase tendra una función que realizara el cambio de dominio de RGB al dominio de profundidad antes de entrar la imagen a un segmentador.

class area(imagen):
    def __init__(self,path_configuracion,nombre_configuracion):
      
        imagen.__init__(self,path_configuracion,nombre_configuracion)
        self.fx=self.depth_intrinseca[0,0]
        self.fy=self.depth_intrinseca[1,1]
        self.constante_area=1/(self.fx*self.fy)
    def abrir_imgs(self,path, indicador1,indicador2,nombre):
        #abrir imagenes
        img2_depth=0
        img_depth=0
        ret,img_rgb,img_depth=self.abrir_par_img(path,indicador1,indicador2,nombre)
       
        if ret:
            #quitar distorsion z
            img_depth=self.distorsion_z(img_depth,filtrar=True)
            #quitar distorsion
            img_rgb,img_depth=self.distorsion(img_rgb,img_depth)
            # cambiar de dominio
            img2_depth=self.rgb2_depth(img_rgb)
        return ret,img2_depth,img_depth
    def calcular_area(self,img,mascara):
        img_z=cv2.bitwise_and(img,img,mask=mascara)
        img_z=np.array(img_z/10,dtype=np.float64)
        #print("distancia maxima de profundidad en mm: {}".format(img_z.max()))
        img_z2=img_z*img_z
        img_area=self.constante_area*img_z2
        area=np.sum(img_area)
        return area
    def calcular_error(self,valor_real,valor_medido):
        e=100*(valor_real-valor_medido)/valor_real
        return e
#################################################################################################################################################
# 1. el constructur de la clase resive el directorio y el nombre del archivo de configuración para cargar todos los datos de calibración
# 2. tiene una función preprocesar_imgs la cual se encarga de resivir la imagen RGB, la imagen de Depth y una función para segmentar. esta función quita la distorsión a ambas imagenes lleva la información RGB al dominio Depth y la segmetan con ayuda de la función entregada.
# 3. calcular_volumen: resive  resive la imagen Depth, la imagen mascara que contine la segementación del objeto y una imagen de referencia depth la cual es el fondo sin objeto alguno,
# 4. la clase tiene funciones adicionales para abrir la imagen de referencia, mejorarla, asi como tambien una función para calcular el error absoluto porcentual de una medida.       
class volumen(imagen):
    def __init__(self,path_configuracion,nombre_configuracion):
      
        imagen.__init__(self,path_configuracion,nombre_configuracion)
        self.fx=self.depth_intrinseca[0,0]
        self.fy=self.depth_intrinseca[1,1]
        self.constante_area=1/(self.fx*self.fy)
       
 
    def preprocesar_imgs(self,img_rgb,img_depth,f_segmentador):
        #quitar distorsion z
        img_depth=self.distorsion_z(img_depth,filtrar=True)
     
        #quitar distorsion uv
        img_rgb,img_depth=self.distorsion(img_rgb,img_depth)
        # cambiar de dominio rgb a depth
        img2_depth=self.rgb2_depth(img_rgb)
        # encontrar mascara
        ## hay que corregir 
        mascara,img=f_segmentador(img2_depth)
        return img_depth,mascara
    def calcular_volumen(self,img,mascara,img_ref):
       
        img_z=cv2.bitwise_and(img,img,mask=mascara)
        img_ref=cv2.bitwise_and(img_ref,img_ref,mask=mascara)

        img_z=np.array(img_z,dtype=np.float64)

        objeto=img_z.copy()[img_z>0].reshape(-1)
        ref=img_ref.copy()[img_ref>0].reshape(-1)
        i_objeto=range(objeto.size)
        i_ref=range(ref.size)
        #plot(list_x=[i_objeto,i_ref],list_y=[objeto,ref],filas=1,colum=2,list_titulos=["altura objeto","altura referencia"],ejes=[["px index","profundidad"],["px index","profundidad"]])
        img_z2=img_z.copy()*img_z.copy()
        
        img_area=self.constante_area*img_z2

        img_altura=img_ref.copy()-img_z.copy()
        
        img_volumen=img_area*img_altura
        volumen=np.sum(img_volumen)
        return volumen
    def calcular_error(self,valor_real,valor_medido):
        e=100*(valor_real-valor_medido)/valor_real
        return e
    def mejorar_referencia(self,img_ref):

        img_ref[img_ref==0]=img_ref.mean()
        img_ref=self.distorsion_z(img_ref,1)
        return img_ref
    def abrir_img_ref(self,path,nombre):
        path_ref=os.path.join(path, '{}.png'.format(nombre))
      
        ret=cv2.haveImageReader(path_ref)
        img_ref=0
        if ret:
            img_ref=cv2.imread(path_ref,cv2.IMREAD_UNCHANGED)[:,:,0]
            img_ref=self.mejorar_referencia(img_ref)
        return ret,img_ref