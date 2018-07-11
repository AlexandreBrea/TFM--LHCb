#TASKS 3
## you use the tracks defined in the previous step to construct all possible muons reaching 4 stations. You discard those muon hits
## then you go back to station three, and do the same, trying to find all remaining muons. You discard those muon hits. Study the efficiency to select KsMuMu muons (those coming from an offline reconstructed downdown Ks->mumu) by selecting muons found so far
## finally you go to station 2. Try to find all muons with hits only in stations 2 and 1. Since you'll have plenty of solutions, you can use the pointing to the 0,0,0 as criteria...


#PROBLEMAS
#Para a eficiencia so reconstruo os que tenhen 4 hits en Mu e non os de 3.








#! /usr/bin/env python
from Configurables import DaVinci, MeasurementProvider
from Configurables import FTRawBankDecoder
from Configurables import DaVinci
import GaudiPython
import os
from PhysSelPython.Wrappers import DataOnDemand
from Configurables import CombineParticles, ChargedProtoParticleMaker, NoPIDsParticleMaker,DaVinci,ChargedPP2MC, LoKi__VertexFitter
from CommonParticles import StdAllNoPIDsMuons
from CommonParticles.Utils import *
from Gaudi.Configuration import NTupleSvc,GaudiSequencer
from Gaudi.Configuration import *
#from Bender.MainMC import *
#! /usr/bin/env python
#from SomeUtils.alyabar import *
#from LinkerInstances.eventassoc import *
#import BenderTools.TisTos
from ROOT import *
import math as m
import sys
import numpy as np
from sklearn.linear_model import LinearRegression

#from BenderAlgo.select import selectVertexMin

from easygraphs import *
import matplotlib.pyplot as plt
#from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D
import ROOT
#from sklearn import datasets, linear_model
#from sklearn.metrics import mean_squared_error, r2_score
c_light = 299.792458
print 'NEW XERAL,100000'
##CREATE ALGORITHM 
algorithm =  NoPIDsParticleMaker ( 'StdNoPIDsDownMuons'  ,
                                DecayDescriptor = 'Muon' ,
                                Particle = 'muons' )

# configure the track selector
selector = trackSelector ( algorithm,
                           trackTypes = ['Downstream'],
                           cuts = { "Chi2Cut" : [0,10] } )

## configure Data-On-Demand service 
locations = updateDoD ( algorithm )

## finally: define the symbol 
StdNoPIDsDownMuons = algorithm 



## MCTRUTH MATCHING
myprotos = ChargedProtoParticleMaker("MyProtoParticles",
                       Inputs = ["Rec/Track/Best"],
                       Output = "Rec/ProtoP/MyProtoParticles")

protop_locations = [myprotos.Output]
charged = ChargedPP2MC("myprotos")
charged.InputData = protop_locations
#ChargedPP2MC().InputData = protop_locations
myseq = GaudiSequencer("myseq")
#myseq.Members +=[myprotos,ChargedPP2MC()]
myseq.Members +=[myprotos,charged]
DaVinci().UserAlgorithms+=[myseq]

##
def get_mcpar(proto):
    LinkRef = GaudiPython.gbl.LHCb.LinkReference()
    linker = TES["Link/Rec/ProtoP/MyProtoParticles/PP2MC"]
    ok = linker.firstReference(proto.key(), None ,LinkRef)
    if not ok: return 0
    return TES["MC/Particles"][LinkRef.objectKey()]


## DOWN DOWN KS0
MyMuonsDown = DataOnDemand(Location = 'Phys/StdNoPIDsDownMuons')
Ks2MuMuDown = CombineParticles("MCSel_Ks2MuMuDown")
Ks2MuMuDown.Preambulo=["from LoKiPhysMC.decorators import *",
                       "from LoKiPhysMC.functions import mcMatch"]
## build KS0->pipi
Ks2MuMuDown.DecayDescriptor = "KS0 -> mu+ mu-"
## only select real KS0s, matched to MCTruth muons
Ks2MuMuDown.DaughtersCuts = {"mu+"  : " mcMatch( '[mu+]cc' )" }
Ks2MuMuDown.MotherCut = " mcMatch('KS0 ==>  mu+ mu-' )"
Ks2MuMuDown.Inputs =['Phys/StdNoPIDsDownMuons']
DaVinci().UserAlgorithms +=[Ks2MuMuDown] ## downstream pions




DaVinci().EvtMax = 0
DaVinci().DataType = "Upgrade"
DaVinci().Simulation = True
DaVinci().DDDBtag  = "upgrade/dddb-20171126"
DaVinci().CondDBtag = "upgrade/sim-20171127-vc-md100"

## CONFIGURATION FOR UPGRADE
MeasurementProvider().IgnoreVelo = True
MeasurementProvider().IgnoreTT = True
MeasurementProvider().IgnoreIT = True
MeasurementProvider().IgnoreOT = True

HOME = "/eos/lhcb/wg/RD/K0S2mu2/Upgrade/ldst/Sim09-Up02/MagDown/"
#DaVinci().Input = [HOME+str(sys.argv[1])]


DaVinci().Input = map(lambda x: 'PFN:root://eoslhcb.cern.ch/'+HOME+x,os.listdir(HOME))
gaudi = GaudiPython.AppMgr()
gaudi.initialize()

TES = gaudi.evtsvc()
MuonDigit = GaudiPython.gbl.LHCb.MuonDigit
LHCbID = GaudiPython.gbl.LHCb.LHCbID

LHCbState = GaudiPython.gbl.LHCb.State
T_extrapolator = gaudi.toolsvc().create("TrackMasterExtrapolator",interface="ITrackExtrapolator")

z1 = 7620 ## pos onde comezan as Tstations
## nace a particula antes das stations?
def overt_cond(par):
    return par.originVertex().position().z()<z1

z2 = 9439 ## pos onde rematan as Tstations
## morre a particula despois das Tstations?
def evert_cond(par):
    ev = map(lambda x: x,par.endVertices())
    ev = ev[-1]
    return ev.position().z()>z2
def define_state(par):
    mystate = LHCbState()
    mystate.setX(par.originVertex().position().x())
    mystate.setY(par.originVertex().position().y())
    mystate.setZ(par.originVertex().position().z())
    mystate.setQOverP(par.particleID().threeCharge()/(3.*par.p()))
    mystate.setTx(par.momentum().x()/par.momentum().z())
    mystate.setTy(par.momentum().y()/par.momentum().z())    
    return mystate

## Extrapolar o estado ata o z procurado, determinar o x e y correspondentes
def get_xy(state,z):
    mystate = state.clone()
    T_extrapolator.propagate(mystate,z)
    return mystate.x(),mystate.y()


def ismm(ks0):
        evs = map(lambda x: x,ks0.endVertices())
        if not len(evs): return False
        return map(lambda x: abs(x.particleID().pid()),evs[-1].products()).count(13)==2

Data={}
DataReco={}
X=[]
k=[]
hits=[]
corruptos=0
numerador=0
denominador=0
CHI2x=[]
CHI2y=[]
sigmafalsex=[]
sigmafalsey=[]
sigmatruex=[]
sigmatruey=[]
sigmas=[0.25,0.5,0.75,1.,1.25,1.5,1.75,2.,2.25,2.5,2.75,3.,3.25,3.5,3.75,4.0,4.25,4.5,4.75,5.0,5.25,5.5,5.75,6.0]
sigmas=[0.125,0.25,0.375,0.5,0.625,0.75,0.875,1.,1.125,1.25,1.375,1.5,1.625,1.75,1.875,2.,2.125,2.25,2.375,2.5,2.625,2.75,2.875,3.]
sigmasT=[0.125,0.25,0.375,0.5,0.625,0.75,0.875,1.,1.125,1.25,1.375,1.5,1.625,1.75,1.875,2.,2.125,2.25,2.375,2.5,2.625,2.75,2.875,3.]
Trues=[0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
TruesReco=[0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
fantasmasReco=[0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
falses=[0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
AllTrues=[0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
fantasmas=[0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
T_measProvider = gaudi.toolsvc().create( "MeasurementProvider",interface="IMeasurementProvider")
#gaudi.run(70)
total=0


survivalstotal=[]
survivals2total=[]
survivals3total=[]
eficienciatotal=[]
numeradoreficiencia=[]

for z in range(len(sigmas)):
	survivalstotal.append([])
	survivals2total.append([])
	survivals3total.append([])
	eficienciatotal.append([])
	numeradoreficiencia.append([])
for z in range(len(sigmas)):
   for zz in range(len(sigmas)):

	survivalstotal[z].append([0])
	survivals2total[z].append([0])
	survivals3total[z].append([0])
	eficienciatotal[z].append([0])
	numeradoreficiencia.append([0])

#print survivals
contador=0
model = LinearRegression()


def SigmaCalculator(un,dous,err):
	return abs((un-dous))/np.sqrt(err**2)

for i in range(300000):
    #contadoreff=0#i=i+1
    print i
    event=i
    gaudi.run(1)
    #if event==1148: continue
    #if event==1149: continue
    Datas={}
    #numeradoreficiencia=0
    if not TES["MC/Particles"]:
                corruptos+=1
                continue
  

   # print T3

    
    tracks = TES["Rec/Track/Best"]
        #ks0s = TES["Phys/MCSel"+str(self.name())+"_Ks2PiPi/Particles"]
    myprotos = TES["Rec/ProtoP/MyProtoParticles"]

    ks0s = TES["Phys/MCSel_Ks2MuMuDown/Particles"]
    tracks = filter(lambda x: x.type()==5,tracks)
    pvs = TES["Rec/Vertex/Primary"]
    #pvs_ = self.vselect("pvs_", ISPRIMARY)
    #if not pvs_.size(): continue
    #ips2cuter = MIPCHI2(pvs_,self.geo())
    #if ks0s:
     #   print '-------------------KS0-------------------'
    if not ks0s:
	print 'no ks0'
	continue
    if ks0s.size()==0: continue
    if not pvs: 
	print 'no pvs'
	continue
    KeysReco=[]
    for ks0 in ks0s:
            if ks0.daughters()[0].particleID().pid()==13: muplus, muminus = ks0.daughters()[0], ks0.daughters()[1]
            else : muminus, muplus = ks0.daughters()[0], ks0.daughters()[1]
	    if not muminus.proto().muonPID() or not muplus.proto().muonPID(): continue	
	    if muminus.proto().muonPID().IsMuon()==False: continue
	    if muplus.proto().muonPID().IsMuon()==False: continue
#	    print 'pt', muplus.pt().value()
#	    print 'pt', muminus.pt().value()
	    KeysReco.append(get_mcpar(muplus.proto()).key())
	    KeysReco.append(get_mcpar(muminus.proto()).key())
	    #PVips2 = VIPCHI2( ks0, self.geo())
            #PV = selectVertexMin(pvs_, PVips2, (PVips2 >= 0. ))
	    #if not PV: continue
    
    if len(KeysReco)==0: continue
    survivals=[]
    survivals2=[]
    survivals3=[]
    eficiencia=[]
    for z in range(len(sigmas)):
        survivals.append([])
        survivals2.append([])
        survivals3.append([])
        eficiencia.append([])
    for z in range(len(sigmas)):
      for zz in range(len(sigmas)):

        survivals[z].append([0])
        survivals2[z].append([0])
        survivals3[z].append([0])
  	eficiencia[z].append([0])
    
    
## coller as particulas que poden atravesar as T stations

    mypars = filter(lambda x: x.particleID().threeCharge()!=0 and overt_cond(x) and evert_cond(x),TES["MC/Particles"])
## converter estas particulas en estados

    for par in mypars:
        if par.momentum().z()==0:
                mypars.remove(par)

    total1=0
    total2=0
    total3=0
    mystates = map(define_state,mypars)
## determinar o x, y de todas estas particulas no que queiramos, neste caso z = 7948 m. Con isto construimos os nosos pseudohits
    myxypos = map(lambda x: get_xy(x,7948),mystates)
    #print len(myxypos)
## collemos os hits dentro da aceptancia
    myxypos = filter(lambda x: abs(x[0])<2417 and abs(x[1])<2417,myxypos)
    total1= len(myxypos)
    #print myxypos
    T1=[]
    for m in range(len(myxypos)):
        T1.append([myxypos[m][0],myxypos[m][1],7948])
   # print T1
#facemos o mesmo para T2 e T3
    z2=8630
    myxyposT2 = map(lambda x: get_xy(x,8630),mystates)
    #print len(myxyposT2)
    myxyposT2 = filter(lambda x: abs(x[0])<2417 and abs(x[1])<2417,myxyposT2)
    total2=len(myxyposT2)
    T2=[]
    for mm in range(len(myxyposT2)):
        T2.append([myxyposT2[mm][0],myxyposT2[mm][1],8630])


    z3=9315
    myxyposT3 = map(lambda x: get_xy(x,9315),mystates)
    #print len(myxyposT3)
    myxyposT3 = filter(lambda x: abs(x[0])<2417 and abs(x[1])<2417,myxyposT3)
    total3 = len(myxyposT3)
    T3=[]
    for mmm in range(len(myxyposT3)):
        T3.append([myxyposT3[mmm][0],myxyposT3[mmm][1],9315])

    denominadoreficiencia=len(KeysReco)
 #   print 'KeysReco', KeysReco
    ## find out KsMuMu candidates
    #ks0s = filter(lambda y: ismm(y),filter(lambda x: x.particleID().pid()==310,TES["MC/Particles"]))
    
    mus=filter(lambda x: abs(x.particleID().pid())==13,TES["MC/Particles"])
    #if len(ks0s): pass
    #else: continue
    #mus = map(lambda x: x.target(),ks0s[0].endVertices()[0].products())
    #mus = filter(lambda x: x.particleID().pid()!=22,mus)
    Keys = map(lambda z: z.key(), mus)
  #  print 'Keys', Keys
    
    ## get muon hits
    coords = map(lambda x: x,TES['Raw/Muon/Coords'])
    ## relation between muon hits and mcparticles
    linkd = TES['Link/Raw/Muon/Digits']
    keys = linkd.keyIndex()
    keysd = {}
    for ke in keys: keysd[ke[0]]=ke[1]
    #aaX=[]
    #aaY=[]
    #bbX=[]
    #bbY=[]
    Hits=[]
    P2=[]
    P1=[]
    P3=[]
    P4=[]
    ## a coord is a crossing of two tiles (first tile is x, second is y)
    for j in range(len(coords)):
        coord = coords[j]
	aaX=[]
        aaY=[]
        bbX=[]
        bbY=[]
	ccX=[]
	ccY=[]
	ddX=[]
	ddY=[]

	#P3=[]
       	#P4=[]
	#print len(coord.digitTile())
	if len(coord.digitTile())!=2:continue
        for tile in coord.digitTile():
            digit = MuonDigit(tile)
            if not (digit.index() in keysd): continue
            mylkey = keysd[digit.index()]
            mcpark = linkd.linkReference()[mylkey].objectKey()
            mcpar = TES["MC/Particles"][mcpark]
	    if "particleID" in dir(mcpar):
		    #if mcpar.particleID() and abs(mcpar.particleID().pid())!=13: continue
			if mcpar.particleID(): pass
			else: continue
            if "key" in dir(mcpar):
		    #if mcpar.key() in Keys: pass
			if mcpar.key():pass
			else: continue
	#	    else: continue
	    else: continue
	    #print 'we are here'
            #print mcpar.key(),mcpar.particleID().pid(),tile.station() ## station can be 1 to 4 (station 0 goes away in the upgrade)
            #T_measProvider = gaudi.toolsvc().create( "MeasurementProvider",interface="IMeasurementProvider")
	    meas = T_measProvider.measurement(LHCbID(tile),0)
	    meas2=T_measProvider.measurement(LHCbID(tile),1)
            #print meas.measure(),meas.errMeasure(),meas.z()
	    #print meas2.measure(),meas2.errMeasure(),meas2.z() 
	    Hits.append(tile.station())
	    #print Hits
	    x=mcpar.key()
	    for j in range(len(KeysReco)):
			if x==KeysReco[j]:
			    #print 'keys coincidence'
                            finalstation=-1
                            if tile.station()>finalstation and 0L in Hits and 1L in Hits and 2L in Hits:
                                    finalstation=tile.station()
                                    Datas[x]=[finalstation]

	    DataReco[i]=Datas	   

            for j in range(len(Keys)):
                        if x==Keys[j]:
                            #print 'keys coincidence'
                            finalstation=-1
                            if tile.station()>finalstation and 0L in Hits and 1L in Hits and 2L in Hits:
                                    finalstation=tile.station()
                                    Data[x]=[finalstation]


	   
	    #print DataReco
	    X1=[]
	    Y1=[]
	    X2=[]
	    Y2=[]	
            X3=[]
            Y3=[]
	    X4=[]
	    Y4=[]

	    if tile.station()==3:
		    
		    if meas.errMeasure()<meas2.errMeasure():
			    X4=[meas.measure(),meas.errMeasure(),meas.z(),mcpar.key(),mcpar.particleID().pid()]
		    else:
			    Y4=[meas2.measure(),meas2.errMeasure(),meas2.z(),mcpar.key(),mcpar.particleID().pid()]
		    if len(X4)>0:
	 		aaX.append(X4)
			AaX=X4
		    else:	
			aaY.append(Y4)
			AaY=Y4
            if tile.station()==2:

		    if meas.errMeasure()<meas2.errMeasure():
			    X3=[meas.measure(),meas.errMeasure(),meas.z(),mcpar.key(),mcpar.particleID().pid()]
		    else:
			    Y3=[meas2.measure(),meas2.errMeasure(),meas2.z(),mcpar.key(),mcpar.particleID().pid()]

		    if len(X3)>0:
	 		bbX.append(X3)
			BbX=X3
		    else:	
			bbY.append(Y3)
			#print Y3
			BbY=Y3

	

            if tile.station()==1:

                    if meas.errMeasure()<meas2.errMeasure():
                            X2=[meas.measure(),meas.errMeasure(),meas.z(),mcpar.key(),mcpar.particleID().pid()]
                    else:
                            Y2=[meas2.measure(),meas2.errMeasure(),meas2.z(),mcpar.key(),mcpar.particleID().pid()]

                    if len(X2)>0:
                        ccX.append(X2)
                        CcX=X2
                    else:
                        ccY.append(Y2)
                        CcY=Y2

            if tile.station()==0:

                    if meas.errMeasure()<meas2.errMeasure():
                            X1=[meas.measure(),meas.errMeasure(),meas.z(),mcpar.key(),mcpar.particleID().pid()]
                    else:
                            Y1=[meas2.measure(),meas2.errMeasure(),meas2.z(),mcpar.key(),mcpar.particleID().pid()]

                    if len(X1)>0:
                        ddX.append(X1)
                        DdX=X1
                    else:
                        ddY.append(Y1)
                        DdY=Y1
       
	if len(bbX)!=0 and len(bbY)!=0:
		P3.append([BbX,BbY])
	if len(aaX)!=0:
		P4.append([AaX,AaY])

	if len(ccX)!=0:
		P2.append([CcX,CcY])
	if len(ddX)!=0:
		P1.append([DdX,DdY])


    p1=[]
    p2=[]
    p3=[]
    p4=[]
    for data in Datas.keys():
	total+=1
    for el in P3:
	p3.append([el[0][0],el[1][0],(el[0][2]+el[1][2])/2,el[0][1],el[1][1],el[0][3],el[0][4]])
    for il in P4:
 	p4.append([il[0][0],il[1][0],(il[0][2]+il[1][2])/2,il[0][1],il[1][1],il[0][3],il[0][4]])	
	
    for ol in P2:
        p2.append([ol[0][0],ol[1][0],(ol[0][2]+ol[1][2])/2,ol[0][1],ol[1][1],ol[0][3],ol[0][4]])
    for ul in P1:
        p1.append([ul[0][0],ul[1][0],(ul[0][2]+ul[1][2])/2,ul[0][1],ul[1][1],ul[0][3],ul[0][4]])


	#g1=graph(z1,x1,z1err,x1err)
        #r1=g1.Fit("pol1","S")
        #chi21=r1.Chi2()       
    #print 'p1:',p1,'p2:',p2,'p3:',p3,'p4:',p4
    #p=[x,y,z,errx,erry]
    candidatesP3=[]
    candidatesP4=[]
    u=[]
    for ell in p4:
	for ill in p3:
		x0=ill[0]
		ux=ell[0]-ill[0]
		y0=ill[1]
		uy=ell[1]-ill[1]
		z0=ill[2]
		uz=ell[2]-ill[2]
		Lambda=(8000.-z0)/uz
#		#print Lambda

		x=x0+Lambda*ux
		y=y0+Lambda*uy

		if abs(x)<3000 and abs(y)<3000:
			candidatesP3.append(ill)
			u.append([ux,uy,uz,np.sqrt(ell[3]**2+ill[3]**2),np.sqrt(ell[4]**2+ill[4]**2)])
			candidatesP4.append(ell)
			   

   # print candidatesP3,candidatesP4	
   # print u	
   # print len(candidatesP3),len(candidatesP4)
    FINALCANDIDATESp4=[]
    FINALCANDIDATESp3=[]
    FINALCANDIDATESp2=[]
    FINALCANDIDATESp1=[]
    #print len(p4),len(p3),len(p2),len(p1)
    for i in range(len(candidatesP3)):
	
   	for ull in p2:
		hitposx=ull[0]
	  	hitposy=ull[1]
		hitposz=ull[2]
		errhitx=ull[3]
		errhity=ull[4]
		Lambda=(hitposz-candidatesP3[i][2])/u[i][2]
		extrapolx=candidatesP3[i][0]+Lambda*u[i][0]
		extrapoly=candidatesP3[i][1]+Lambda*u[i][1]
		#errextrapol=np.sqrt(errx0**2+u1**2*err(lambda)**2+lambda**2*err(u1)**2
		
		#errLambda=np.sqrt((hitposz-candidatesP3[i][2])**2*(1.0/(u[i][2])**2)**2*0)
		#errLambda=0
		errextrapolx=np.sqrt(candidatesP3[i][3]**2+(Lambda**2*(u[i][3])**2))
		errextrapoly=np.sqrt(candidatesP3[i][4]**2+(Lambda**2*(u[i][4])**2))
		Sigmax2=(extrapolx-hitposx)/np.sqrt(errhitx**2+errextrapolx**2)
		Sigmay2=(extrapoly-hitposy)/np.sqrt(errhity**2+errextrapoly**2)
		#print 'Sigma',Sigmax2,Sigmay2

		if abs(Sigmax2)<6.0 and abs(Sigmay2)<6.0:	
		     for oll in p1:
	                hitposx=oll[0]
        	        hitposy=oll[1]
                	hitposz=oll[2]
               		errhitx=oll[3]
                	errhity=oll[4]
               		Lambda=(hitposz-candidatesP3[i][2])/u[i][2]
                	extrapolx=candidatesP3[i][0]+Lambda*u[i][0]
                	extrapoly=candidatesP3[i][1]+Lambda*u[i][1]
                	errextrapolx=np.sqrt(candidatesP3[i][3]**2+(Lambda**2*(u[i][3])**2))
                	errextrapoly=np.sqrt(candidatesP3[i][4]**2+(Lambda**2*(u[i][4])**2))
                	Sigmax=(extrapolx-hitposx)/np.sqrt(errhitx**2+errextrapolx**2)
                	Sigmay=(extrapoly-hitposy)/np.sqrt(errhity**2+errextrapoly**2)
			
			if abs(Sigmax)<6.0 and abs(Sigmay)<6.0:
				FINALCANDIDATESp4.append([candidatesP4[i],[abs(Sigmax),abs(Sigmay),abs(Sigmax2),abs(Sigmay2)]])
				FINALCANDIDATESp3.append([candidatesP3[i],[abs(Sigmax),abs(Sigmay),abs(Sigmax2),abs(Sigmay2)]])
				FINALCANDIDATESp2.append([ull,[abs(Sigmax),abs(Sigmay),abs(Sigmax2),abs(Sigmay2)]])
				FINALCANDIDATESp1.append([oll,[abs(Sigmax),abs(Sigmay),abs(Sigmax2),abs(Sigmay2)]])

  #  print 'FINALCANDIDATES4',FINALCANDIDATESp4,len(FINALCANDIDATESp4),'FINALCANDIDATES3',FINALCANDIDATESp3,'FINALCANDIDATES2',FINALCANDIDATESp2,'FINALCANDIDATES1',FINALCANDIDATESp1
    def POP(FINALCANDIDATESp4,FINALCANDIDATESp3,FINALCANDIDATESp2,FINALCANDIDATESp1): 
     if len(FINALCANDIDATESp4)>1:
      for j in range(len(FINALCANDIDATESp4)-1):
	k=j+1
	#print k
	#print len(FINALCANDIDATESp4),len(FINALCANDIDATESp3),len(FINALCANDIDATESp2),len(FINALCANDIDATESp1)
	if k>(len(FINALCANDIDATESp4)-1): break
	if FINALCANDIDATESp4[k][0][0]==FINALCANDIDATESp4[k-1][0][0] and FINALCANDIDATESp3[k][0][0]==FINALCANDIDATESp3[k-1][0][0]:
		if FINALCANDIDATESp1[k][0][0]!=FINALCANDIDATESp1[k-1][0][0]:
			if np.sqrt(FINALCANDIDATESp1[k][1][0]**2+FINALCANDIDATESp1[k][1][1]**2) > np.sqrt(FINALCANDIDATESp1[k-1][1][0]**2+FINALCANDIDATESp1[k-1][1][1]**2):
				FINALCANDIDATESp4.pop(k)
				FINALCANDIDATESp3.pop(k)
				FINALCANDIDATESp2.pop(k)
				FINALCANDIDATESp1.pop(k)
				k=k+1
				if len(FINALCANDIDATESp4)>1: pass
				else: break
	
			else:
				FINALCANDIDATESp4.pop(k-1)
                                FINALCANDIDATESp3.pop(k-1)
                                FINALCANDIDATESp2.pop(k-1)
                                FINALCANDIDATESp1.pop(k-1)
				k=k+1
				if len(FINALCANDIDATESp4)>1: pass
				else: break
		
		if k>(len(FINALCANDIDATESp4)-1): break

		if FINALCANDIDATESp2[k][0][0]!=FINALCANDIDATESp2[k-1][0][0]:
                        if np.sqrt(FINALCANDIDATESp2[k][1][2]**2+FINALCANDIDATESp2[k][1][3]**2) > np.sqrt(FINALCANDIDATESp2[k-1][1][2]**2+FINALCANDIDATESp2[k-1][1][3]**2):
                                FINALCANDIDATESp4.pop(k)
                                FINALCANDIDATESp3.pop(k)
                                FINALCANDIDATESp2.pop(k)
                                FINALCANDIDATESp1.pop(k)
				if len(FINALCANDIDATESp4)>1: pass
                                else: break

                        else:
                                FINALCANDIDATESp4.pop(k-1)
                                FINALCANDIDATESp3.pop(k-1)
                                FINALCANDIDATESp2.pop(k-1)
                                FINALCANDIDATESp1.pop(k-1)
				if len(FINALCANDIDATESp4)>1: pass
                                else: break
     return FINALCANDIDATESp4, FINALCANDIDATESp3, FINALCANDIDATESp2, FINALCANDIDATESp1
    
    oki=0
    if len(FINALCANDIDATESp4)>1:
     while oki==0:
        FINALCANDIDATESp4,FINALCANDIDATESp3,FINALCANDIDATESp2,FINALCANDIDATESp1=POP(FINALCANDIDATESp4,FINALCANDIDATESp3,FINALCANDIDATESp2,FINALCANDIDATESp1)
        if len(FINALCANDIDATESp4)>1:
	 for jj in range(len(FINALCANDIDATESp4)-1):
	  if FINALCANDIDATESp4[jj]==FINALCANDIDATESp4[jj+1] and FINALCANDIDATESp3[jj]==FINALCANDIDATESp3[jj+1] and FINALCANDIDATESp2[jj]==FINALCANDIDATESp2[jj+1]  and FINALCANDIDATESp1[jj]==FINALCANDIDATESp1[jj+1]: 
		oki=0
		FINALCANDIDATESp4.pop(jj)
		FINALCANDIDATESp3.pop(jj)
		FINALCANDIDATESp2.pop(jj)
		FINALCANDIDATESp1.pop(jj)
		break
	  else: oki=1
        else: oki=1

    P4def=[]
    P3def=[]
    P2def=[]
    P1def=[]
    finalkeys4=[]
    for k in range(len(sigmas)):
	P4def.append([])
	P3def.append([])
	P2def.append([])
	P1def.append([])
	finalkeys4.append([])
    #print 'F',FINALCANDIDATESp4
 #   print sigmas
    for jjjj in range(len(sigmas)):
        #finalkeys4=[]
#	print jjjj
        for ii in range(len(FINALCANDIDATESp4)):
            #aqui FALTAN AS OUTRAS SIGMAS!
            if abs(FINALCANDIDATESp1[ii][1][0])<sigmas[jjjj] and abs(FINALCANDIDATESp1[ii][1][1])<sigmas[jjjj] and abs(FINALCANDIDATESp2[ii][1][2])<sigmas[jjjj] and abs(FINALCANDIDATESp2[ii][1][3])<sigmas[jjjj]: 
                if FINALCANDIDATESp3[ii][0][-2]==FINALCANDIDATESp2[ii][0][-2] and FINALCANDIDATESp2[ii][0][-2]==FINALCANDIDATESp1[ii][0][-2] and FINALCANDIDATESp4[ii][0][-2]==FINALCANDIDATESp3[ii][0][-2] and abs(FINALCANDIDATESp3[ii][0][-1])==13:
                        #print SECONDFINALCANDIDATESp1[ii][0][-1]
                        AllTrues[jjjj]=AllTrues[jjjj]+1
			#print FINALCANDIDATESp4[ii][0][-2]
                      #  print FINALCANDIDATESp3[ii][0][-2]
                      #  print FINALCANDIDATESp2[ii][0][-2]
                      #  print FINALCANDIDATESp1[ii][0][-2]

                        if FINALCANDIDATESp1[ii][0][-2] in finalkeys4[jjjj]:

                                 fantasmas[jjjj]=fantasmas[jjjj]+1
                        else:

				if FINALCANDIDATESp3[ii][0][-2] in finalkeys4[jjjj-1]:
				  for q in range(len(P4def[jjjj-1])):	
				    if FINALCANDIDATESp4[ii]==P4def[jjjj-1][q] or FINALCANDIDATESp3[ii]==P3def[jjjj-1][q] or FINALCANDIDATESp2[ii]==P2def[jjjj-1][q] or FINALCANDIDATESp1[ii]==P1def[jjjj-1][q]:
			#		print 'THIS HAPPEN'
					P4def[jjjj].append(P4def[jjjj-1][q])
                                        P3def[jjjj].append(P3def[jjjj-1][q])
                                        P2def[jjjj].append(P2def[jjjj-1][q])
                                        P1def[jjjj].append(P1def[jjjj-1][q])
 	


                                else:
				    if FINALCANDIDATESp4[ii][0][-2] in Datas.keys():
                                        finalkeys4[jjjj].append(FINALCANDIDATESp1[ii][0][-2])
                                        TruesReco[jjjj]=TruesReco[jjjj]+1
                                        Trues[jjjj]=Trues[jjjj]+1
	                               
					P4def[jjjj].append(FINALCANDIDATESp4[ii])
					P3def[jjjj].append(FINALCANDIDATESp3[ii])
					P2def[jjjj].append(FINALCANDIDATESp2[ii])
					P1def[jjjj].append(FINALCANDIDATESp1[ii])

					

                                        #print FINALCANDIDATESp4[ii]
                                        #print FINALCANDIDATESp3[ii]
                                        #print FINALCANDIDATESp2[ii]
                                        #print FINALCANDIDATESp1[ii]
                                    else:
                                        #fantasmas[j]=fantasmas[j]+1
                                        Trues[jjjj]=Trues[jjjj]+1
                else: fantasmas[jjjj]=fantasmas[jjjj]+1
            else: falses[jjjj]=falses[jjjj]+1
    #print P4def
    
    for element in FINALCANDIDATESp4:
	if element[0] in p4:
	#	print 'p4delete'
		p4.remove(element[0])

    for element3 in FINALCANDIDATESp3:
        if element3[0] in p3:
         #       print 'p3delete'
                p3.remove(element3[0])

    for element2 in FINALCANDIDATESp2:
        if element2[0] in p2:
          #      print 'p2delete'
                p2.remove(element2[0])

    for element1 in FINALCANDIDATESp1:
	if element1[0] in p1:
	#	print 'p1delete'
		p1.remove(element1[0])



    secondcandidatesP3=[]
    secondcandidatesP2=[]
    v=[]
    for ell in p3:
        for ill in p2:
                x0=ill[0]
                ux=ell[0]-ill[0]
                y0=ill[1]
                uy=ell[1]-ill[1]
                z0=ill[2]
                uz=ell[2]-ill[2]
                Lambda=(8000.-z0)/uz
#               #print Lambda

                x=x0+Lambda*ux
                y=y0+Lambda*uy

                if abs(x)<3000 and abs(y)<3000:
                        secondcandidatesP2.append(ill)
                        v.append([ux,uy,uz,np.sqrt(ell[3]**2+ill[3]**2),np.sqrt(ell[4]**2+ill[4]**2)])
                        secondcandidatesP3.append(ell)



    SECONDFINALCANDIDATESp3=[]
    SECONDFINALCANDIDATESp2=[]
    SECONDFINALCANDIDATESp1=[]

    for i in range(len(secondcandidatesP3)):

        for ull in p1:
                hitposx=ull[0]
                hitposy=ull[1]
                hitposz=ull[2]
                errhitx=ull[3]
                errhity=ull[4]
                Lambda=(hitposz-secondcandidatesP2[i][2])/v[i][2]
                extrapolx=secondcandidatesP2[i][0]+Lambda*v[i][0]
                extrapoly=secondcandidatesP2[i][1]+Lambda*v[i][1]
                #errextrapol=np.sqrt(errx0**2+u1**2*err(lambda)**2+lambda**2*err(u1)**2

                #errLambda=np.sqrt((hitposz-candidatesP3[i][2])**2*(1.0/(u[i][2])**2)**2*0)
                #errLambda=0
                errextrapolx=np.sqrt(secondcandidatesP2[i][3]**2+(Lambda**2*(v[i][3])**2))
                errextrapoly=np.sqrt(secondcandidatesP2[i][4]**2+(Lambda**2*(v[i][4])**2))
                Sigmax2=(extrapolx-hitposx)/np.sqrt(errhitx**2+errextrapolx**2)
                Sigmay2=(extrapoly-hitposy)/np.sqrt(errhity**2+errextrapoly**2)
                #print 'Sigma',Sigmax2,Sigmay2

                if abs(Sigmax2)<6.0 and abs(Sigmay2)<6.0:
                                SECONDFINALCANDIDATESp3.append([secondcandidatesP3[i],[abs(Sigmax2),abs(Sigmay2)]])
                                SECONDFINALCANDIDATESp2.append([secondcandidatesP2[i],[abs(Sigmax2),abs(Sigmay2)]])
                                SECONDFINALCANDIDATESp1.append([ull,[abs(Sigmax2),abs(Sigmay2)]])

    #print 'SECOND', SECONDFINALCANDIDATESp3,'SECOND 2',SECONDFINALCANDIDATESp2,'SECOND 1', SECONDFINALCANDIDATESp1

    def POP2(SECONDFINALCANDIDATESp3,SECONDFINALCANDIDATESp2,SECONDFINALCANDIDATESp1):
     if len(SECONDFINALCANDIDATESp3)>1:
      for j in range(len(SECONDFINALCANDIDATESp3)-1):
        k=j+1
        #print k
        #len(SECONDFINALCANDIDATESp3),len(SECONDFINALCANDIDATESp2),len(SECONDFINALCANDIDATESp1)
        if k>(len(SECONDFINALCANDIDATESp3)-1): break
        if SECONDFINALCANDIDATESp3[k][0][0]==SECONDFINALCANDIDATESp3[k-1][0][0] and SECONDFINALCANDIDATESp2[k][0][0]==SECONDFINALCANDIDATESp2[k-1][0][0]:
                if SECONDFINALCANDIDATESp1[k][0][0]!=SECONDFINALCANDIDATESp1[k-1][0][0]:
                        if np.sqrt(SECONDFINALCANDIDATESp1[k][1][0]**2+SECONDFINALCANDIDATESp1[k][1][1]**2) > np.sqrt(SECONDFINALCANDIDATESp1[k-1][1][0]**2+SECONDFINALCANDIDATESp1[k-1][1][1]**2):
                               
                                SECONDFINALCANDIDATESp3.pop(k)
                                SECONDFINALCANDIDATESp2.pop(k)
                                SECONDFINALCANDIDATESp1.pop(k)
                                k=k+1
                                if len(SECONDFINALCANDIDATESp3)>1: pass
                                else: break

                        else:
                                
                                SECONDFINALCANDIDATESp3.pop(k-1)
                                SECONDFINALCANDIDATESp2.pop(k-1)
                                SECONDFINALCANDIDATESp1.pop(k-1)
                                k=k+1
                                if len(SECONDFINALCANDIDATESp3)>1: pass
                                else: break

                if k>(len(SECONDFINALCANDIDATESp3)-1): break

     return SECONDFINALCANDIDATESp3, SECONDFINALCANDIDATESp2, SECONDFINALCANDIDATESp1

    oki=0
    if len(SECONDFINALCANDIDATESp3)>0: continue
     #while oki==0:
      #  SECONDFINALCANDIDATESp3,SECONDFINALCANDIDATESp2,SECONDFINALCANDIDATESp1=POP2(SECONDFINALCANDIDATESp3,SECONDFINALCANDIDATESp2,SECONDFINALCANDIDATESp1)
       # if len(SECONDFINALCANDIDATESp3)>1:
        # for jj in range(len(SECONDFINALCANDIDATESp3)-1):
         # if SECONDFINALCANDIDATESp3[jj]==SECONDFINALCANDIDATESp3[jj+1] and SECONDFINALCANDIDATESp2[jj]==SECONDFINALCANDIDATESp2[jj+1] and SECONDFINALCANDIDATESp1[jj]==SECONDFINALCANDIDATESp1[jj+1]: 
           #     oki=0            
          #      SECONDFINALCANDIDATESp3.pop(jj)
            #    SECONDFINALCANDIDATESp2.pop(jj)
             #   SECONDFINALCANDIDATESp1.pop(jj)
              #  break
         # else: oki=1
        #else: oki=1
       
   # print 'SECOND', SECONDFINALCANDIDATESp3,'SECOND 2',SECONDFINALCANDIDATESp2,'SECOND 1', SECONDFINALCANDIDATESp1

    #sigmas=[0.5,1.,1.5,2.,2.5,3.]    
    for i in range(len(SECONDFINALCANDIDATESp3)):
	if SECONDFINALCANDIDATESp3[i][0][-2]==SECONDFINALCANDIDATESp2[i][0][-2] and SECONDFINALCANDIDATESp2[i][0][-2]==SECONDFINALCANDIDATESp1[i][0][-2] and abs(SECONDFINALCANDIDATESp3[i][0][-1])==13:
		sigmatruex.append(SECONDFINALCANDIDATESp1[i][1][0])
		sigmatruey.append(SECONDFINALCANDIDATESp1[i][1][1])
	else:
		sigmafalsex.append(SECONDFINALCANDIDATESp1[i][1][0])
		sigmafalsey.append(SECONDFINALCANDIDATESp1[i][1][1])

   

    p3DEF=[]
    p2DEF=[]
    p1DEF=[]
    finalkeys=[]
    for vv in range(len(sigmas)):
	p3DEF.append([])
    	p2DEF.append([])
	p1DEF.append([])
 	finalkeys.append([])
    station4=[]
    for key in Datas.keys():
    	if Datas[key]==[3L]:
		station4.append(key)
    #print station4
    for j in range(len(sigmas)):
	#finalkeys=[]
        for ii in range(len(SECONDFINALCANDIDATESp3)):
	    #print 'a',SECONDFINALCANDIDATESp3[ii][0][-2]
	
	#for j in range(len(sigmas))
	    if SECONDFINALCANDIDATESp3[ii][0][-2] in station4 or SECONDFINALCANDIDATESp2[ii][0][-2] in station4 or SECONDFINALCANDIDATESp1[ii][0][-2] in station4: continue

	    if abs(SECONDFINALCANDIDATESp1[ii][1][0])<sigmas[j] and abs(SECONDFINALCANDIDATESp1[ii][1][1])<sigmas[j]:
		if SECONDFINALCANDIDATESp3[ii][0][-2]==SECONDFINALCANDIDATESp2[ii][0][-2] and SECONDFINALCANDIDATESp2[ii][0][-2]==SECONDFINALCANDIDATESp1[ii][0][-2] and abs(SECONDFINALCANDIDATESp3[ii][0][-1])==13:
			#print SECONDFINALCANDIDATESp1[ii][0][-1]
			AllTrues[j]=AllTrues[j]+1
			if SECONDFINALCANDIDATESp1[ii][0][-2] in finalkeys[j]:
				 
				 fantasmas[j]=fantasmas[j]+1
			else:
				#if SECONDFINALCANDIDATESp3[ii][0][-2] in finalkeys[j-1]:
                                  #for q in range(len(p3DEF[j-1])):
                                    #if SECONDFINALCANDIDATESp3[ii]==p3DEF[j-1][q] or SECONDFINALCANDIDATESp2[ii]==p2DEF[j-1][q] or SECONDFINALCANDIDATESp1[ii]==p1DEF[j-1][q]:
                                        
                                       # p3DEF[j].append(p3DEF[j-1][q])
                                       # p2DEF[j].append(p2DEF[j-1][q])
                                       # p1DEF[j].append(p1DEF[j-1][q])


				
				#else:
				    if SECONDFINALCANDIDATESp3[ii][0][-2] in Datas.keys():	 
					finalkeys[j].append(SECONDFINALCANDIDATESp1[ii][0][-2])
					TruesReco[j]=TruesReco[j]+1
					Trues[j]=Trues[j]+1
					p3DEF[j]=SECONDFINALCANDIDATESp3[ii]
					p2DEF[j]=SECONDFINALCANDIDATESp2[ii]
					p1DEF[j]=SECONDFINALCANDIDATESp1[ii]

				    else: 
					#fantasmas[j]=fantasmas[j]+1
					Trues[j]=Trues[j]+1
		else: fantasmas[j]=fantasmas[j]+1
	    else: falses[j]=falses[j]+1

   
    #agora vou crear listas cos puntos para axustar duas rectas por cada muon reconstruido (xz) e (yz)

    #myhits son os que quedan en cada caso, sendo o total total1,2,3
    myhits=[]
    xT1=[]
    xT2=[]
    xT3=[]
    yT1=[]
    yT2=[]
    yT3=[]

    for i in range(len(sigmas)):    
		xT1.append([])
		xT2.append([])
    		xT3.append([])
    		yT1.append([])
    		yT2.append([])
    		yT3.append([])

    for j in range(len(P4def)):
	#print j
        myhits.append([])
	for jj in range(len(P4def[j])):
		myhits.append([])
#		print jj

		x4=P4def[j][jj][0][0]
		y4=P4def[j][jj][0][1]
		z4=P4def[j][jj][0][2]
		errx4=P4def[j][jj][0][3]
		erry4=P4def[j][jj][0][4]


		x3=P3def[j][jj][0][0]
                y3=P3def[j][jj][0][1]
                z3=P3def[j][jj][0][2]
                errx3=P3def[j][jj][0][3]
                erry3=P3def[j][jj][0][4]

		x2=P2def[j][jj][0][0]
                y2=P2def[j][jj][0][1]
                z2=P2def[j][jj][0][2]
                errx2=P2def[j][jj][0][3]
                erry2=P2def[j][jj][0][4]

		x1=P1def[j][jj][0][0]
                y1=P1def[j][jj][0][1]
                z1=P1def[j][jj][0][2]
                errx1=P1def[j][jj][0][3]
                erry1=P1def[j][jj][0][4]

		xx= [x4,x3,x2,x1]
		errxx=[errx4,errx3,errx2,errx1]
		zz =[z4,z3,z2,z1]
		errzz=[0,0,0,0]
		yy= [y4,y3,y2,y1]
		erryy=[erry4,erry3,erry2,erry1]
		#xx=np.array(xx)
		#zz=np.array(zz)
#		print xx,zz,errxx,errzz
		g1=graph(zz,xx,errzz,errxx)
                r1=g1.Fit("pol1","S")
		
#  xx = a + b*zz

                erra=r1.Errors()[0]
		errb=r1.Errors()[1]
		
		a=r1.GetParams()[0]
		b=r1.GetParams()[1]
		
#	        print a,b
		XT1=b*7948+a
		XT2=b*8630+a
		XT3=b*9315+a
		#print 'root', XT1,XT2,XT3
		#now s(x)=np.sqrt((z*s(a))**2+s(b)**2)
		sx1=np.sqrt((7948*errb)**2+erra**2)
		sx2=np.sqrt((8630*errb)**2+erra**2)
		sx3=np.sqrt((9315*errb)**2+erra**2)
                #chi21=r1.Chi2()

		#model = LinearRegression()
		#model.fit(zz, xx)
		#model.get_params()
    		#xT1[j].append(model.predict(7948))

		xT1[j].append([XT1,sx1])
		xT2[j].append([XT2,sx2])
		xT3[j].append([XT3,sx3])
		

                #yT1[j].append(model.predict(7948))
		#yT2[j].append(model.predict(8630))
		#yT3[j].append(model.predict(9315))
		
		g2=graph(zz,yy,errzz,erryy)
                r2=g2.Fit("pol1","S")

#  xx = a + b*zz

                erra2=r2.Errors()[0]
                errb2=r2.Errors()[1]

                a2=r2.GetParams()[0]
                b2=r2.GetParams()[1]
                YT1=b2*7948+a2
                YT2=b2*8630+a2
                YT3=b2*9315+a2
                #print 'root', XT1,XT2,XT3
                #now s(x)=np.sqrt((z*s(a))**2+s(b)**2)
                sy1=np.sqrt((7948*errb2)**2+erra2**2)
                sy2=np.sqrt((8630*errb2)**2+erra2**2)
                sy3=np.sqrt((9315*errb2)**2+erra2**2)
		
        	yT1[j].append([YT1,sy1])
                yT2[j].append([YT2,sy2])
                yT3[j].append([YT3,sy3])

    if len(xT1)==0: continue
    #print 'xT1',xT1
    #print 'xT2',xT2
    #print 'xT3',xT3

    #print 'yT1',yT1
    #print 'yT2',yT2
    #print 'yT3',yT3

    x3T1=[]
    x3T2=[]
    x3T3=[]
    y3T1=[]
    y3T2=[]
    y3T3=[]
 #   print total1,total2,total3
    for i in range(len(sigmas)):    
		x3T1.append([])
		x3T2.append([])
    		x3T3.append([])
    		y3T1.append([])
    		y3T2.append([])
    		y3T3.append([])




#    if len(p3DEF)!=0:
 #     for j in range(len(p3DEF)):
        #myhits.append([])
#	if len(p3DEF[j])!=0:
#	    for jj in range(len(p3DEF[j])):
#		#myhits.append([])
#		print p3DEF[j]
#		print jj
#		print p3DEF[j][jj]
#		if len(p3DEF[j][jj][0])==0:continue
#		x3=p3DEF[j][jj][0][0]
 #               y3=p3DEF[j][jj][0][1]
  #              z3=p3DEF[j][jj][0][2]
   #             errx3=p3DEF[j][jj][0][3]
   #             erry3=p3DEF[j][jj][0][4]

#		x2=p2DEF[j][jj][0][0]
    #            y2=p2DEF[j][jj][0][1]
 #               z2=p2DEF[j][jj][0][2]
  #              errx2=p2DEF[j][jj][0][3]
   #             erry2=p2DEF[j][jj][0][4]
#
#		x1=p1DEF[j][jj][0][0]
 #               y1=p1DEF[j][jj][0][1]
  #              z1=p1DEF[j][jj][0][2]
   #             errx1=p1DEF[j][jj][0][3]
    #            erry1=p1DEF[j][jj][0][4]

#		xx= [x3,x2,x1]
#		errxx=[errx3,errx2,errx1]
#		zz =[z3,z2,z1]
#		errzz=[0,0,0]
#		yy= [y3,y2,y1]
#		erryy=[erry3,erry2,erry1]
		#xx=np.array(xx)
		#zz=np.array(zz)
	#	print xx,zz
#		g1=graph(zz,xx,errzz,errxx)
 #               r1=g1.Fit("pol1","S")
		
#  xx = a + b*zz

 #               erra=r1.Errors()[0]
#		errb=r1.Errors()[1]
		
#		a=r1.GetParams()[0]
#		b=r1.GetParams()[1]
		
	        #print a,b
#		XT1=b*7948+a
#		XT2=b*8630+a
#		XT3=b*9315+a
		#print 'root', XT1,XT2,XT3
		#now s(x)=np.sqrt((z*s(a))**2+s(b)**2)
#		sx1=np.sqrt((7948*errb)**2+erra**2)
#		sx2=np.sqrt((8630*errb)**2+erra**2)
#		sx3=np.sqrt((9315*errb)**2+erra**2)
                #chi21=r1.Chi2()

		#model = LinearRegression()
		#model.fit(zz, xx)
		#model.get_params()
    		#xT1[j].append(model.predict(7948))

#		x3T1[j].append([XT1,sx1])
#		x3T2[j].append([XT2,sx2])
#		x3T3[j].append([XT3,sx3])
		

                #yT1[j].append(model.predict(7948))
		#yT2[j].append(model.predict(8630))
		#yT3[j].append(model.predict(9315))
		
#		g2=graph(zz,yy,errzz,erryy)
 #               r2=g2.Fit("pol1","S")

#  xx = a + b*zz

  #              erra2=r2.Errors()[0]
   #             errb2=r2.Errors()[1]

     #           a2=r2.GetParams()[0]
    #            b2=r2.GetParams()[1]
      #          YT1=b2*7948+a2
       #         YT2=b2*8630+a2
        #        YT3=b2*9315+a2
                #print 'root', XT1,XT2,XT3
                #now s(x)=np.sqrt((z*s(a))**2+s(b)**2)
         #       sy1=np.sqrt((7948*errb2)**2+erra2**2)
          #      sy2=np.sqrt((8630*errb2)**2+erra2**2)
           #     sy3=np.sqrt((9315*errb2)**2+erra2**2)
		
        #	y3T1[j].append([YT1,sy1])
         #       y3T2[j].append([YT2,sy2])
          #      y3T3[j].append([YT3,sy3])

    contadoreff=[]
    contadoreff2=[]
    contadoreff3=[]
    m=0
    for i in range(len(sigmas)):
	if len(xT1[i])>m: m=len(xT1[i])
    for i in range(len(sigmas)):
	contadoreff.append([])
	contadoreff2.append([])
        contadoreff3.append([])
    for i in range(len(sigmas)):
	for ii in range(len(sigmas)):
 		contadoreff[i].append([])
    		contadoreff2[i].append([])
		contadoreff3[i].append([])
    for j in range(len(sigmas)):
	for jj in range(len(sigmas)):
	    for i in range(m):
	       	contadoreff[j][jj].append(0)
		contadoreff2[j][jj].append(0)
		contadoreff3[j][jj].append(0)
    if len(xT1)!=0:
		contador+=1
    #print 'len', m
    
    #print len(xT1),len(sigmas)
    #print 'contador',contador
    #print contadoreff
    for i in range(len(sigmas)):
	for ii in range(len(sigmas)):
          for el in T1:
		
		if len(xT1)!=0:
			if  len(xT1[ii])!=0:
			    #for j in range(len(xT1[ii])):
				 
                              for j in range(len(xT1[ii])):

                                 #print xT1[ii]
                                 #print j
                                 if SigmaCalculator(el[0],xT1[ii][j][0],xT1[ii][j][1])<sigmasT[i] and SigmaCalculator(el[1],yT1[ii][j][0],yT1[ii][j][1])<sigmasT[i] :
                                        #print survivals[0]
                                        survivals[i][ii][0]+=float(1)/total1
                                        #non me faria falta o numeradoreficiencia
 				#	print contadoreff[i][ii]
                                        contadoreff[i][ii][j]+=1

				

    #print 'CONTADOR',contadoreff
					
    #a condicion que preciso sera contadoreff=len(xT1[j])
    for i in range(len(sigmas)):
        for ii in range(len(sigmas)):
          for el in T2:
                #print el
                if len(xT2)!=0:
                        if  len(xT2[ii])!=0:
  			   for j in range(len(xT2[ii])):
                                 #print xT1[ii]
                                 if SigmaCalculator(el[0],xT2[ii][j][0],xT2[ii][j][1])<sigmasT[i] and SigmaCalculator(el[1],yT2[ii][j][0],yT2[ii][j][1])<sigmasT[i] :
                                        #print survivals[0]
                                        survivals2[i][ii][0]+=float(1)/total2
					contadoreff2[i][ii][j]+=1




    for i in range(len(sigmas)):
        for ii in range(len(sigmas)):
          for el in T3:
                #print el
                if len(xT3)!=0:
                        if  len(xT3[ii])!=0:
			     for j in range(len(xT3[ii])):
                                 #print xT1[ii]
                                 if SigmaCalculator(el[0],xT3[ii][j][0],xT3[ii][j][1])<sigmasT[i] and SigmaCalculator(el[1],yT3[ii][j][0],yT3[ii][j][1])<sigmasT[i] :
                                        #print survivals[0]
                                        survivals3[i][ii][0]+=float(1)/total3
					contadoreff3[i][ii][j]+=1



    #print '------------------------'
    #print survivals
    #print '------------------------'
    #print survivals2
    #print '------------------------'
    #print survivals3



    for i in range(len(sigmas)):
        for ii in range(len(sigmas)):
                        survivalstotal[i][ii][0]+=survivals[i][ii][0]

    for i in range(len(sigmas)):
        for ii in range(len(sigmas)):
                        survivals2total[i][ii][0]+=survivals2[i][ii][0]
    for i in range(len(sigmas)):
        for ii in range(len(sigmas)):
                        survivals3total[i][ii][0]+=survivals3[i][ii][0]

    #print survivalstotal

    for i in range(len(sigmas)):
	for ii in range(len(sigmas)):
		for j in range(len(contadoreff[i][ii])):
                        c=0
			if contadoreff[i][ii][j]!=0 and contadoreff2[i][ii][j]!=0 and contadoreff3[i][ii][j]!=0:
		
				eficiencia[i][ii][0]+=float(1)/denominadoreficiencia

  #  print 'eff', eficiencia	

    for i in range(len(sigmas)):
        for ii in range(len(sigmas)):
                        eficienciatotal[i][ii][0]+=eficiencia[i][ii][0]

   # print 'efftotal', eficienciatotal

print 'contador', contador
print 'survivals', survivalstotal
print 'eff',eficienciatotal
for i in range(len(sigmas)):
	for ii in range(len(sigmas)):
		survivalstotal[i][ii]=survivalstotal[i][ii][0]/contador

for i in range(len(sigmas)):
        for ii in range(len(sigmas)):
                survivals2total[i][ii]=survivals2total[i][ii][0]/contador

for i in range(len(sigmas)):
        for ii in range(len(sigmas)):
                survivals3total[i][ii]=survivals3total[i][ii][0]/contador



for i in range(len(sigmas)):
        for ii in range(len(sigmas)):
                eficienciatotal[i][ii]=eficienciatotal[i][ii]/contador


print 'survivalsdef', survivalstotal
print 'effdef',efftotal

