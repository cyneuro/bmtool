:Interneuron Cells to Pyramidal Cells GABA with local Ca2+ pool and read public soma Ca2+ pool

NEURON {
	POINT_PROCESS interD2interD_STFD
	USEION ca READ eca,ica
	NONSPECIFIC_CURRENT igaba
	RANGE initW
	RANGE Cdur_gaba, AlphaTmax_gaba, Beta_gaba, Erev_gaba, gbar_gaba, W, on_gaba, g_gaba
	RANGE eca, tauCa, Icatotal
	RANGE ICag, P0g, fCag
	RANGE Cainf, pooldiam, z
	RANGE lambda1, lambda2, threshold1, threshold2
	RANGE fmax, fmin, Wmax, Wmin, maxChange, normW, scaleW, srcid, destid,limitW,irecord
	RANGE pregid,postgid, thr_rp
	RANGE F, f, tauF, D1, d1, tauD1, D2, d2, tauD2
	RANGE facfactor
    RANGE neuroM,type
}

UNITS {
	(mV) = (millivolt)
        (nA) = (nanoamp)
	(uS) = (microsiemens)
	FARADAY = 96485 (coul)
	pi = 3.141592 (1)
}

PARAMETER {

	srcid = -1 (1)
	destid = -1 (1)
	type = -1
	
	Cdur_gaba = 0.7254 (ms)
	AlphaTmax_gaba = 1.853 (/ms):7.2609 (/ms): 2.2609 (/ms): 3.2609 (/ms)   : 7.2609 as original
	Beta_gaba = 0.147(/ms) : 0.147 (/ms) : 0.2667 (/ms):         : 0.2667 as original
	Erev_gaba = -75 (mV) : -75 as original
	gbar_gaba = 0.2e-3 (uS) : 0.6e-3 (uS)

	Cainf = 50e-6 (mM)
	pooldiam =  1.8172 (micrometer)
	z = 2

    neuroM = 0
	k = 0.01	
	
	tauCa = 50 (ms)
	
	P0g = .01
	fCag = .024
	
	lambda1 = 1 : 0.7 : 0.6 : 0.7 : 1.0 : 0.5 : 1.5 :3 : 4 : 3 : 2 : 3.0 : 2.0
	lambda2 = .01
	threshold1 = 0.5  :0.47 :  0.48 : 0.45 : 0.4 : 0.95 : 1.35 :0.75 :0.55 (uM)
	threshold2 = 0.6 :0.52 :  0.53 : 0.5 : 0.45 : 1.0 : 1.4 : 0.8 : 0.65 :0.70 (uM)

	:GABA Weight
	initW = 2.0 : 5.0 : 3.0 : 4.0 : 5.0 : 4.2 : 3.5 :4.5 : :  :  3 :  2.5 : 5
	fmax = 3 : 2.85 :4 : 3 : 3
	fmin = .8
	
	GAPstart1 = 96000 
	GAPstop1 = 196000
	
	thr_rp = 1 : .7
	
	facfactor = 1
	: the (1) is needed for the range limits to be effective
        f = 0 (1) < 0, 1e9 > : 1.3 (1) < 0, 1e9 >    : facilitation
        tauF = 20 (ms) < 1e-9, 1e9 >
        d1 = 0.95 (1) < 0, 1 >     : fast depression
        tauD1 = 40 (ms) < 1e-9, 1e9 >
        d2 = 0.9 (1) < 0, 1 >     : slow depression
        tauD2 = 70 (ms) < 1e-9, 1e9 >	
	
    DAstart1 = 39500
	DAstop1 = 40000	
	DAstart2 = 35900
	DAstop2 = 36000	

	DA_t1 = 0.7 : 0.7
	DA_t2 = 1.5 : 1.3 : 1.2
	DA_t3 = 1.25
	DA_S = 1.6 : 1.8 : 1.8					
	Beta1 = 0.001  (/ms) : 1/decay time for neuromodulators
	Beta2 = 0.0001  (/ms)	
	
	NEstart1 = 39500
	NEstop1 = 40000	
	NEstart2 = 35900
	NEstop2 = 36000		


	NE_t1 = 1 : 1 : 0.95

	NE_t2 = 1 :1 : 0.7 : 0.8
	NE_t3 = 1
	NE_S = 1 : 0.4
}

ASSIGNED {
	v (mV)
	eca (mV)
	ica (nA)
	
	igaba (nA)
	g_gaba (uS)
	on_gaba
    irecord (nA)
	
	limitW

	t0 (ms)

	ICan (nA)
	ICag (nA)
	Afactor	(mM/ms/nA)
	Icatotal (nA)

	dW_gaba
	Wmax
	Wmin
	maxChange
	normW
	scaleW
	
	pregid
	postgid

	rp
	tsyn
	
	fa
	F
	D1
	D2	
}

STATE { r_nmda r_gaba capoolcon W }

INITIAL {

	on_gaba = 0
	r_gaba = 0
	W = initW
	limitW = 1

	t0 = -1

	Wmax = fmax*initW
	Wmin = fmin*initW
	maxChange = (Wmax-Wmin)/10
	dW_gaba = 0

	capoolcon = Cainf
	Afactor	= 1/(z*FARADAY*4/3*pi*(pooldiam/2)^3)*(1e6)

	fa =0
	F = 1
	D1 = 1
	D2 = 1	
}

BREAKPOINT {
if ((eta(capoolcon)*(lambda1*omega(capoolcon, threshold1, threshold2)-lambda2*GAP1(GAPstart1, GAPstop1)*W))>0&&W>=Wmax) {
        limitW=1e-12
	} else if ((eta(capoolcon)*(lambda1*omega(capoolcon, threshold1, threshold2)-lambda2*GAP1(GAPstart1, GAPstop1)*W))<0&&W<=Wmin) {
        limitW=1e-12
	} else {
	limitW=1 }
	
	SOLVE release METHOD cnexp
	 : if (W >= Wmax || W <= Wmin ) {     : for limiting the weight
	 : limitW=1e-12
	 : } else {
	  : limitW=1
	 : }
	 :if (W > Wmax) { 
		:W = Wmax
	:} else if (W < Wmin) {
 		:W = Wmin
	:}
	 
	    if (neuroM==1) {
	g_gaba = gbar_gaba*r_gaba*facfactor*DA1(DAstart1,DAstop1)*DA2(DAstart2,DAstop2)   : Dopamine effect on GABA	
	} else if (neuroM==2) {
	g_gaba = gbar_gaba*r_gaba*facfactor*NEn(NEstart1,NEstop1)*NE2(NEstart2,NEstop2)   : Norepinephrine effect on GABA		    	
	} else if (neuroM==3) {
	g_gaba = gbar_gaba*r_gaba*facfactor*DA1(DAstart1,DAstop1)*DA2(DAstart2,DAstop2)*NEn(NEstart1,NEstop1)*NE2(NEstart2,NEstop2)   : Dopamine & Norepinephrine effect on GABA		    
	} else {
	g_gaba = gbar_gaba*r_gaba*facfactor
	}

    igaba = W*g_gaba*(v - Erev_gaba)
    irecord=igaba
	ICag = P0g*g_gaba*(v - eca)	
	Icatotal = ICag + k*ica*4*pi*((15/2)^2)*(0.01)    :  icag+k*ica*Area of soma*unit change

	
}

DERIVATIVE release {
    
	: W' = eta(capoolcon)*(lambda1*omega(capoolcon, threshold1, threshold2)-lambda2*GAP1(GAPstart1, GAPstop1)*W)	  : Long-term plasticity was implemented. (Shouval et al. 2002a, 2002b)
    
	
	W' = 1e-12*limitW*eta(capoolcon)*(lambda1*omega(capoolcon, threshold1, threshold2)-lambda2*GAP1(GAPstart1, GAPstop1)*W)
	
	r_gaba' = AlphaTmax_gaba*on_gaba*(1-r_gaba)-Beta_gaba*r_gaba
    capoolcon'= -fCag*Afactor*Icatotal + (Cainf-capoolcon)/tauCa		
}

NET_RECEIVE(dummy_weight) {
      if (flag==0) {           :a spike arrived, start onset state if not already on
         if ((!on_gaba)){       :this synpase joins the set of synapses in onset state
           t0=t
	      on_gaba=1		
	      net_send(Cdur_gaba,1)  
         } else if (on_gaba==1) {             :already in onset state, so move offset time
          net_move(t+Cdur_gaba)
		  t0=t
	      }
         }		  
	if (flag == 1) { : turn off transmitter, i.e. this synapse enters the offset state	
	on_gaba=0
    }
	
if (flag == 0) {  : Short term plasticity was implemented(Varela et. al 1997):
	rp = unirand()	
	
	:F  = 1 + (F-1)* exp(-(t - tsyn)/tauF)
	D1 = 1 - (1-D1)*exp(-(t - tsyn)/tauD1)
	D2 = 1 - (1-D2)*exp(-(t - tsyn)/tauD2)
 :printf("%g\t%g\t%g\t%g\t%g\t%g\n", t, t-tsyn, F, D1, D2, facfactor)
	:printf("%g\t%g\t%g\t%g\n", F, D1, D2, facfactor)
	tsyn = t
	
	facfactor = F * D1 * D2

	::F = F+f  :F * f
	
	if (F > 3) { 
	F=3	}	
	if (facfactor < 0.7) { 
	facfactor=0.7
	}
	D1 = D1 * d1
	D2 = D2 * d2
:printf("\t%g\t%g\t%g\n", F, D1, D2)
}
}

:::::::::::: FUNCTIONs and PROCEDUREs ::::::::::::

FUNCTION eta(Cani (mM)) {
	LOCAL taulearn, P1, P2, P4, Cacon
	P1 = 0.1
	P2 = P1*1e-4
	P4 = 1
	Cacon = Cani*1e3
	taulearn = P1/(P2+Cacon*Cacon*Cacon)+P4
	eta = 1/taulearn*0.001
}

FUNCTION omega(Cani (mM), threshold1 (uM), threshold2 (uM)) {
	LOCAL r, mid, Cacon
	Cacon = Cani*1e3
	r = (threshold2-threshold1)/2
	mid = (threshold1+threshold2)/2
	if (Cacon <= threshold1) { omega = 0}
	else if (Cacon >= threshold2) {	omega = 1/(1+50*exp(-50*(Cacon-threshold2)))}
	else {omega = -sqrt(r*r-(Cacon-mid)*(Cacon-mid))}
}
FUNCTION DA1(DAstart1 (ms), DAstop1 (ms)) {
	LOCAL DAtemp1, DAtemp2, DAtemp3, DAtemp4, DAtemp5, DAtemp6, DAtemp7, DAtemp8, DAtemp9, DAtemp10, DAtemp11, DAtemp12, DAtemp13, DAtemp14, DAtemp15, DAtemp16, DAtemp17, DAtemp18, DAtemp19, DAtemp20, DAtemp21, DAtemp22, DAtemp23, DAtemp24, DAtemp25, DAtemp26, DAtemp27, DAtemp28, DAtemp29, DAtemp30, DAtemp31, DAtemp32, DAtemp33, DAtemp34,s
	DAtemp1 = DAstart1+4000
	DAtemp2 = DAtemp1+4000
	DAtemp3 = DAtemp2+4000
	DAtemp4 = DAtemp3+4000
	DAtemp5 = DAtemp4+4000
	DAtemp6 = DAtemp5+4000
	DAtemp7 = DAtemp6+4000
	DAtemp8 = DAtemp7+4000
	DAtemp9 = DAtemp8+4000
	DAtemp10 = DAtemp9+4000
	DAtemp11 = DAtemp10+4000
	DAtemp12 = DAtemp11+4000
	DAtemp13 = DAtemp12+4000
	DAtemp14 = DAtemp13+4000
	DAtemp15 = DAtemp14 + 4000 + 100000     : 100sec Gap
	DAtemp16 = DAtemp15 + 4000 
	DAtemp17 = DAtemp16 + 4000
	DAtemp18 = DAtemp17 + 4000
	DAtemp19 = DAtemp18 + 4000 
	DAtemp20 = DAtemp19 + 4000
	DAtemp21 = DAtemp20 + 4000
	DAtemp22 = DAtemp21 + 4000 
	DAtemp23 = DAtemp22 + 4000
	DAtemp24 = DAtemp23 + 4000
	DAtemp25 = DAtemp24 + 4000 
	DAtemp26 = DAtemp25 + 4000
	DAtemp27 = DAtemp26 + 4000
	DAtemp28 = DAtemp27 + 4000 
	DAtemp29 = DAtemp28 + 4000
	DAtemp30 = DAtemp29 + 4000
	DAtemp31 = DAtemp30 + 4000 
	DAtemp32 = DAtemp31 + 4000
	DAtemp33 = DAtemp32 + 4000
	DAtemp34 = DAtemp33 + 4000

	if (t <= DAstart1) { DA1 = 1.0}
	else if (t >= DAstart1 && t <= DAstop1) {DA1 = DA_t1}					: 2nd tone in conditioning
		else if (t > DAstop1 && t < DAtemp1) {DA1 = 1.0 + (DA_t1-1)*exp(-Beta1*(t-DAstop1))}  			: Basal level
	else if (t >= DAtemp1 && t <= DAtemp1+500) {DA1=DA_t1}					: 3rd tone
		else if (t > DAtemp1+500 && t < DAtemp2) {DA1 = 1.0 + (DA_t1-1)*exp(-Beta1*(t-(DAtemp1+500)))} 		: Basal level
	else if (t >= DAtemp2 && t <= DAtemp2+500) {DA1=DA_t1}					: 4th tone
		else if (t > DAtemp2+500 && t < DAtemp3) {DA1 = 1.0 + (DA_t1-1)*exp(-Beta1*(t-(DAtemp2+500)))} 		: Basal level	
	else if (t >= DAtemp3 && t <= DAtemp3+500) {DA1=DA_t1}					: 5th tone
		else if (t > DAtemp3+500 && t < DAtemp4) {DA1 = 1.0 + (DA_t1-1)*exp(-Beta1*(t-(DAtemp3+500)))} 		: Basal level
	else if (t >= DAtemp4 && t <= DAtemp4+500) {DA1=DA_t1}					: 6th tone
		else if (t > DAtemp4+500 && t < DAtemp5) {DA1 = 1.0 + (DA_t1-1)*exp(-Beta1*(t-(DAtemp4+500)))} 		: Basal level
	else if (t >= DAtemp5 && t <= DAtemp5+500) {DA1=DA_t1}					: 7th tone
		else if (t > DAtemp5+500 && t < DAtemp6) {DA1 = 1.0 + (DA_t1-1)*exp(-Beta1*(t-(DAtemp5+500)))} 		: Basal level
	else if (t >= DAtemp6 && t <= DAtemp6+500) {DA1=DA_t1}					: 8th tone
		else if (t > DAtemp6+500 && t < DAtemp7) {DA1 = 1.0 + (DA_t1-1)*exp(-Beta1*(t-(DAtemp6+500)))} 		: Basal level
	else if (t >= DAtemp7 && t <= DAtemp7+500) {DA1=DA_t1}					: 9th tone
		else if (t > DAtemp7+500 && t < DAtemp8) {DA1 = 1.0 + (DA_t1-1)*exp(-Beta1*(t-(DAtemp7+500)))} 		: Basal level
	else if (t >= DAtemp8 && t <= DAtemp8+500) {DA1=DA_t1}					: 10th tone  
		else if (t > DAtemp8+500 && t < DAtemp9) {DA1 = 1.0 + (DA_t1-1)*exp(-Beta1*(t-(DAtemp8+500)))} 		: Basal level
	
	else if (t >= DAtemp9 && t <= DAtemp9+500) {DA1=DA_t2}					: 11th tone   - Second Step
		else if (t > DAtemp9+500 && t < DAtemp10) {DA1 = 1.0 + (DA_t2-1)*exp(-Beta2*(t-(DAtemp9+500)))}		: Basal level	
	else if (t >= DAtemp10 && t <= DAtemp10+500) {DA1=DA_t2}					: 12th tone
		else if (t > DAtemp10+500 && t < DAtemp11) {DA1 = 1.0 + (DA_t2-1)*exp(-Beta2*(t-(DAtemp10+500)))}	: Basal level
	else if (t >= DAtemp11 && t <= DAtemp11+500) {DA1=DA_t2}					: 13th tone
		else if (t > DAtemp11+500 && t < DAtemp12) {DA1 = 1.0 + (DA_t2-1)*exp(-Beta2*(t-(DAtemp11+500)))}	: Basal level
	else if (t >= DAtemp12 && t <= DAtemp12+500) {DA1=DA_t2}					: 14th tone 
		else if (t > DAtemp12+500 && t < DAtemp13) {DA1 = 1.0 + (DA_t2-1)*exp(-Beta2*(t-(DAtemp12+500)))}	: Basal level
	else if (t >= DAtemp13 && t <= DAtemp13+500) {DA1=DA_t2}					: 15th tone
		else if (t > DAtemp13+500 && t < DAtemp14) {DA1 = 1.0 + (DA_t2-1)*exp(-Beta2*(t-(DAtemp13+500)))}	: Basal level
	else if (t >= DAtemp14 && t <= DAtemp14+500) {DA1=DA_t2}					: 16th tone
		else if (t > DAtemp14+500 && t < DAtemp15) {DA1 = 1.0 + (DA_t2-1)*exp(-Beta2*(t-(DAtemp14+500)))} 	: Basal level
	
	else if (t >= DAtemp15 && t <= DAtemp15+500) {DA1 = DA_t2}					: 1st tone EE
		else if (t > DAtemp15+500 && t < DAtemp16) {DA1 = 1.0 + (DA_t2-1)*exp(-Beta2*(t-(DAtemp15+500)))}  	: Basal level
	else if (t >= DAtemp16 && t <= DAtemp16+500) {DA1 = DA_t2}					: 2nd tone EE
		else if (t > DAtemp16+500 && t < DAtemp17) {DA1 = 1.0 + (DA_t2-1)*exp(-Beta2*(t-(DAtemp16+500)))}  	: Basal level
	else if (t >= DAtemp17 && t <= DAtemp17+500) {DA1 = DA_t2}					: 3rd tone EE
		else if (t > DAtemp17+500 && t < DAtemp18) {DA1 = 1.0 + (DA_t2-1)*exp(-Beta2*(t-(DAtemp17+500)))}  	: Basal level	
	else if (t >= DAtemp18 && t <= DAtemp18+500) {DA1 = DA_t2}					: 4th tone EE	
		else if (t > DAtemp18+500 && t < DAtemp19) {DA1 = 1.0 + (DA_t2-1)*exp(-Beta2*(t-(DAtemp18+500)))}  	: Basal level
	else if (t >= DAtemp19 && t <= DAtemp19+500) {DA1 = DA_t3}					: 5th tone EE
		else if (t > DAtemp19+500 && t < DAtemp20) {DA1 = 1.0 + (DA_t2-1)*exp(-Beta2*(t-(DAtemp19+500)))}  	: Basal level
	else if (t >= DAtemp20 && t <= DAtemp20+500) {DA1 = DA_t3}					: 6th tone EE
		else if (t > DAtemp20+500 && t < DAtemp21) {DA1 = 1.0 + (DA_t2-1)*exp(-Beta2*(t-(DAtemp20+500)))}  	: Basal level
	else if (t >= DAtemp21 && t <= DAtemp21+500) {DA1 = DA_t3}					: 7th tone EE
		else if (t > DAtemp21+500 && t < DAtemp22) {DA1 = 1.0 + (DA_t2-1)*exp(-Beta2*(t-(DAtemp21+500)))}  	: Basal level	
	else if (t >= DAtemp22 && t <= DAtemp22+500) {DA1 = DA_t3}					: 8th tone EE	
		else if (t > DAtemp22+500 && t < DAtemp23) {DA1 = 1.0 + (DA_t2-1)*exp(-Beta2*(t-(DAtemp22+500)))}  	: Basal level
	else if (t >= DAtemp23 && t <= DAtemp23+500) {DA1 = DA_t3}					: 9th tone EE
		else if (t > DAtemp23+500 && t < DAtemp24) {DA1 = 1.0 + (DA_t2-1)*exp(-Beta2*(t-(DAtemp23+500)))}  	: Basal level
	else if (t >= DAtemp24 && t <= DAtemp24+500) {DA1 = DA_t3}					: 10th tone EE
		else if (t > DAtemp24+500 && t < DAtemp25) {DA1 = 1.0 + (DA_t2-1)*exp(-Beta2*(t-(DAtemp24+500)))}  	: Basal level
	else if (t >= DAtemp25 && t <= DAtemp25+500) {DA1 = DA_t3}					: 11th tone EE
		else if (t > DAtemp25+500 && t < DAtemp26) {DA1 = 1.0 + (DA_t2-1)*exp(-Beta2*(t-(DAtemp25+500)))}  	: Basal level	
	else if (t >= DAtemp26 && t <= DAtemp26+500) {DA1 = DA_t3}					: 12th tone EE	
		else if (t > DAtemp26+500 && t < DAtemp27) {DA1 = 1.0 + (DA_t2-1)*exp(-Beta2*(t-(DAtemp26+500)))}  	: Basal level
	else if (t >= DAtemp27 && t <= DAtemp27+500) {DA1 = DA_t3}					: 13th tone EE
		else if (t > DAtemp27+500 && t < DAtemp28) {DA1 = 1.0 + (DA_t2-1)*exp(-Beta2*(t-(DAtemp27+500)))}  	: Basal level
	else if (t >= DAtemp28 && t <= DAtemp28+500) {DA1 = DA_t3}					: 14th tone EE
		else if (t > DAtemp28+500 && t < DAtemp29) {DA1 = 1.0 + (DA_t2-1)*exp(-Beta2*(t-(DAtemp28+500)))}  	: Basal level
	else if (t >= DAtemp29 && t <= DAtemp29+500) {DA1 = DA_t3}					: 15th tone EE
		else if (t > DAtemp29+500 && t < DAtemp30) {DA1 = 1.0 + (DA_t2-1)*exp(-Beta2*(t-(DAtemp29+500)))}  	: Basal level	
	else if (t >= DAtemp30 && t <= DAtemp30+500) {DA1 = DA_t3}					: 16th tone EE	
		else if (t > DAtemp30+500 && t < DAtemp31) {DA1 = 1.0 + (DA_t2-1)*exp(-Beta2*(t-(DAtemp30+500)))}  	: Basal level
	else if (t >= DAtemp31 && t <= DAtemp31+500) {DA1 = DA_t3}					: 17th tone EE
		else if (t > DAtemp31+500 && t < DAtemp32) {DA1 = 1.0 + (DA_t2-1)*exp(-Beta2*(t-(DAtemp31+500)))}  	: Basal level
	else if (t >= DAtemp32 && t <= DAtemp32+500) {DA1 = DA_t3}					: 18th tone EE
		else if (t > DAtemp32+500 && t < DAtemp33) {DA1 = 1.0 + (DA_t2-1)*exp(-Beta2*(t-(DAtemp32+500)))}  	: Basal level
	else if (t >= DAtemp33 && t <= DAtemp33+500) {DA1 = DA_t3}					: 19th tone EE
		else if (t > DAtemp33+500 && t < DAtemp34) {DA1 = 1.0 + (DA_t2-1)*exp(-Beta2*(t-(DAtemp33+500)))}  	: Basal level	
	else if (t >= DAtemp34 && t <= DAtemp34+500) {DA1 = DA_t3}					: 20th tone EE		
		else  {	DA1 = 1.0}
}
FUNCTION DA2(DAstart2 (ms), DAstop2 (ms)) {
	LOCAL DA2temp1, DA2temp2, DA2temp3, DA2temp4, DA2temp5, DA2temp6, DA2temp7, DA2temp8, DA2temp9, DA2temp10, DA2temp11, DA2temp12, DA2temp13, DA2temp14, DA2temp15, DA2temp16,s
	DA2temp1 = DAstart2 + 4000
	DA2temp2 = DA2temp1 + 4000
	DA2temp3 = DA2temp2 + 4000
	DA2temp4 = DA2temp3 + 4000
	DA2temp5 = DA2temp4 + 4000
	DA2temp6 = DA2temp5 + 4000
	DA2temp7 = DA2temp6 + 4000
	DA2temp8 = DA2temp7 + 4000
	DA2temp9 = DA2temp8 + 4000
	DA2temp10 = DA2temp9 + 4000
	DA2temp11 = DA2temp10 + 4000
	DA2temp12 = DA2temp11 + 4000 
	DA2temp13 = DA2temp12 + 4000
	DA2temp14 = DA2temp13 + 4000
	DA2temp15 = DA2temp14 + 4000
	
	if (t <= DAstart2) { DA2 = 1.0}
	else if (t >= DAstart2 && t <= DAstop2) {DA2 = DA_S }					: 1st shock
		else if (t > DAstop2 && t < DA2temp1) {DA2 = 1.0 + (DA_S-1)*exp(-Beta2*(t-(DAstop2+500)))}  					 
	else if (t >= DA2temp1 && t <= DA2temp1+100) {DA2=DA_S}					: 2nd shock
		else if (t > DA2temp1+100 && t < DA2temp2) {DA2 = 1.0 + (DA_S-1)*exp(-Beta2*(t-(DA2temp1+100)))}    				 
	else if (t >= DA2temp2 && t <= DA2temp2+100) {DA2=DA_S}					: 3rd shock
		else if (t > DA2temp2+100 && t < DA2temp3) {DA2 = 1.0 + (DA_S-1)*exp(-Beta2*(t-(DA2temp2+100)))}   				 
	else if (t >= DA2temp3 && t <= DA2temp3+100) {DA2=DA_S}					: 4th shock
		else if (t > DA2temp3+100 && t < DA2temp4) {DA2 = 1.0 + (DA_S-1)*exp(-Beta2*(t-(DA2temp3+100)))}   				 
	else if (t >= DA2temp4 && t <= DA2temp4+100) {DA2=DA_S}					: 5th shock
		else if (t > DA2temp4+100 && t < DA2temp5) {DA2 = 1.0 + (DA_S-1)*exp(-Beta2*(t-(DA2temp4+100)))}   				 
	else if (t >= DA2temp5 && t <= DA2temp5+100) {DA2=DA_S}					: 6th shock
		else if (t > DA2temp5+100 && t < DA2temp6) {DA2 = 1.0 + (DA_S-1)*exp(-Beta2*(t-(DA2temp5+100)))}    				 
	else if (t >= DA2temp6 && t <= DA2temp6+100) {DA2=DA_S}					: 7th shock
		else if (t > DA2temp6+100 && t < DA2temp7) {DA2 = 1.0 + (DA_S-1)*exp(-Beta2*(t-(DA2temp6+100)))}   				 
	else if (t >= DA2temp7 && t <= DA2temp7+100) {DA2=DA_S}					: 8th shock
		else if (t > DA2temp7+100 && t < DA2temp8) {DA2 = 1.0 + (DA_S-1)*exp(-Beta2*(t-(DA2temp7+100)))}   				    
	else if (t >= DA2temp8 && t <= DA2temp8+100) {DA2=DA_S }					: 9th shock
		else if (t > DA2temp8+100 && t < DA2temp9) {DA2 = 1.0 + (DA_S-1)*exp(-Beta2*(t-(DA2temp8+100)))}   				    
	else if (t >= DA2temp9 && t <= DA2temp9+100) {DA2=DA_S }					: 10th shock
		else if (t > DA2temp9+100 && t < DA2temp10) {DA2 = 1.0 + (DA_S-1)*exp(-Beta2*(t-(DA2temp9+100)))}   				    
	else if (t >= DA2temp10 && t <= DA2temp10+100) {DA2=DA_S}					: 11th shock
		else if (t > DA2temp10+100 && t < DA2temp11) {DA2 = 1.0 + (DA_S-1)*exp(-Beta2*(t-(DA2temp10+100)))}   				 
	else if (t >= DA2temp11 && t <= DA2temp11+100) {DA2=DA_S }					: 12th shock
		else if (t > DA2temp11+100 && t < DA2temp12) {DA2 = 1.0 + (DA_S-1)*exp(-Beta2*(t-(DA2temp11+100)))}   				 
	else if (t >= DA2temp12 && t <= DA2temp12+100) {DA2=DA_S}					: 13th shock
		else if (t > DA2temp12+100 && t < DA2temp13) {DA2 = 1.0 + (DA_S-1)*exp(-Beta2*(t-(DA2temp12+100)))}   				 
	else if (t >= DA2temp13 && t <= DA2temp13+100) {DA2=DA_S }					: 14th shock
		else if (t > DA2temp13+100 && t < DA2temp14) {DA2 = 1.0 + (DA_S-1)*exp(-Beta2*(t-(DA2temp13+100)))}   				 
	else if (t >= DA2temp14 && t <= DA2temp14+100) {DA2=DA_S}					: 15th shock
		else if (t > DA2temp14+100 && t < DA2temp15) {DA2 = 1.0 + (DA_S-1)*exp(-Beta2*(t-(DA2temp14+100)))}   				 
	else if (t >= DA2temp15 && t <= DA2temp15+100) {DA2=DA_S}					: 16th shock
		else  {	DA2 = 1.0}
}

FUNCTION NEn(NEstart1 (ms), NEstop1 (ms)) {
	LOCAL NEtemp1, NEtemp2, NEtemp3, NEtemp4, NEtemp5, NEtemp6, NEtemp7, NEtemp8, NEtemp9, NEtemp10, NEtemp11, NEtemp12, NEtemp13, NEtemp14, NEtemp15, NEtemp16, NEtemp17, NEtemp18, NEtemp19, NEtemp20, NEtemp21, NEtemp22, NEtemp23, NEtemp24, NEtemp25, NEtemp26, NEtemp27, NEtemp28, NEtemp29, NEtemp30, NEtemp31, NEtemp32, NEtemp33, NEtemp34,s
	NEtemp1 = NEstart1+4000
	NEtemp2 = NEtemp1+4000
	NEtemp3 = NEtemp2+4000
	NEtemp4 = NEtemp3+4000
	NEtemp5 = NEtemp4+4000
	NEtemp6 = NEtemp5+4000
	NEtemp7 = NEtemp6+4000
	NEtemp8 = NEtemp7+4000
	NEtemp9 = NEtemp8+4000
	NEtemp10 = NEtemp9+4000
	NEtemp11 = NEtemp10+4000
	NEtemp12 = NEtemp11+4000
	NEtemp13 = NEtemp12+4000
	NEtemp14 = NEtemp13+4000
	NEtemp15 = NEtemp14 + 4000 + 100000     : 100sec Gap
	NEtemp16 = NEtemp15 + 4000 
	NEtemp17 = NEtemp16 + 4000
	NEtemp18 = NEtemp17 + 4000
	NEtemp19 = NEtemp18 + 4000 
	NEtemp20 = NEtemp19 + 4000
	NEtemp21 = NEtemp20 + 4000
	NEtemp22 = NEtemp21 + 4000 
	NEtemp23 = NEtemp22 + 4000
	NEtemp24 = NEtemp23 + 4000
	NEtemp25 = NEtemp24 + 4000 
	NEtemp26 = NEtemp25 + 4000
	NEtemp27 = NEtemp26 + 4000
	NEtemp28 = NEtemp27 + 4000 
	NEtemp29 = NEtemp28 + 4000
	NEtemp30 = NEtemp29 + 4000
	NEtemp31 = NEtemp30 + 4000 
	NEtemp32 = NEtemp31 + 4000
	NEtemp33 = NEtemp32 + 4000
	NEtemp34 = NEtemp33 + 4000

	if (t <= NEstart1) { NEn = 1.0}
	else if (t >= NEstart1 && t <= NEstop1) {NEn = NE_t1}					: 2nd tone in early conditioning (EC)
		else if (t > NEstop1 && t < NEtemp1) {NEn = 1.0 + (NE_t1-1)*exp(-Beta1*(t-NEstop1))}  		: Basal level
	else if (t >= NEtemp1 && t <= NEtemp1+500) {NEn = NE_t1}					: 3rd tone EC
		else if (t > NEtemp1+500 && t < NEtemp2) {NEn = 1.0 + (NE_t1-1)*exp(-Beta1*(t-NEstop1))}  	: Basal level
	else if (t >= NEtemp2 && t <= NEtemp2+500) {NEn = NE_t1}					: 4th tone EC
		else if (t > NEtemp2+500 && t < NEtemp3) {NEn = 1.0 + (NE_t1-1)*exp(-Beta1*(t-(NEtemp2+500)))}  	: Basal level	
	else if (t >= NEtemp3 && t <= NEtemp3+500) {NEn = NE_t1}					: 5th tone EC
		else if (t > NEtemp3+500 && t < NEtemp4) {NEn = 1.0 + (NE_t1-1)*exp(-Beta1*(t-(NEtemp3+500)))}  	: Basal level
	else if (t >= NEtemp4 && t <= NEtemp4+500) {NEn = NE_t1}					: 6th tone EC
		else if (t > NEtemp4+500 && t < NEtemp5) {NEn = 1.0 + (NE_t1-1)*exp(-Beta1*(t-(NEtemp4+500)))}  		: Basal level
	else if (t >= NEtemp5 && t <= NEtemp5+500) {NEn = NE_t1}					: 7th tone EC
		else if (t > NEtemp5+500 && t < NEtemp6) {NEn = 1.0 + (NE_t1-1)*exp(-Beta1*(t-(NEtemp5+500)))}  	: Basal level
	else if (t >= NEtemp6 && t <= NEtemp6+500) {NEn = NE_t1}					: 8th tone EC
		else if (t > NEtemp6+500 && t < NEtemp7) {NEn = 1.0 + (NE_t1-1)*exp(-Beta1*(t-(NEtemp6+500)))}  	: Basal level
	
	else if (t >= NEtemp7 && t <= NEtemp7+500) {NEn = NE_t2}					: 9th tone	- Second Step late cond (LC)
		else if (t > NEtemp7+500 && t < NEtemp8) {NEn = 1.0 + (NE_t2-1)*exp(-Beta2*(t-(NEtemp7+500)))}  		: Basal level
	else if (t >= NEtemp8 && t <= NEtemp8+500) {NEn = NE_t2}					: 10th tone  LC
		else if (t > NEtemp8+500 && t < NEtemp9) {NEn = 1.0 + (NE_t2-1)*exp(-Beta2*(t-(NEtemp8+500)))}		: Basal level	
	else if (t >= NEtemp9 && t <= NEtemp9+500) {NEn = NE_t2}					: 11th tone  LC 
		else if (t > NEtemp9+500 && t < NEtemp10) {NEn = 1.0 + (NE_t2-1)*exp(-Beta2*(t-(NEtemp9+500)))}  	: Basal level	
	else if (t >= NEtemp10 && t <= NEtemp10+500) {NEn = NE_t2}					: 12th tone  LC
		else if (t > NEtemp10+500 && t < NEtemp11) {NEn = 1.0 + (NE_t2-1)*exp(-Beta2*(t-(NEtemp10+500)))}  	: Basal level
	else if (t >= NEtemp11 && t <= NEtemp11+500) {NEn = NE_t2}					: 13th tone  LC
		else if (t > NEtemp11+500 && t < NEtemp12) {NEn = 1.0 + (NE_t2-1)*exp(-Beta2*(t-(NEtemp11+500)))}  	: Basal level
	else if (t >= NEtemp12 && t <= NEtemp12+500) {NEn = NE_t2}					: 14th tone  LC
		else if (t > NEtemp12+500 && t < NEtemp13) {NEn = 1.0 + (NE_t2-1)*exp(-Beta2*(t-(NEtemp12+500)))}  	: Basal level
	else if (t >= NEtemp13 && t <= NEtemp13+500) {NEn = NE_t2}					: 15th tone  LC
		else if (t > NEtemp13+500 && t < NEtemp14) {NEn = 1.0 + (NE_t2-1)*exp(-Beta2*(t-(NEtemp13+500)))}  	: Basal level
	else if (t >= NEtemp14 && t <= NEtemp14+500) {NEn = NE_t2}					: 16th tone  LC
		else if (t > NEtemp14+500 && t < NEtemp15) {NEn = 1.0 + (NE_t2-1)*exp(-Beta2*(t-(NEtemp14+500)))}  	: Basal level
	
	else if (t >= NEtemp15 && t <= NEtemp15+500) {NEn = NE_t2}					: 1st tone EE
		else if (t > NEtemp15+500 && t < NEtemp16) {NEn = 1.0 + (NE_t2-1)*exp(-Beta2*(t-(NEtemp15+500)))}  	: Basal level
	else if (t >= NEtemp16 && t <= NEtemp16+500) {NEn = NE_t2}					: 2nd tone EE
		else if (t > NEtemp16+500 && t < NEtemp17) {NEn = 1.0 + (NE_t2-1)*exp(-Beta2*(t-(NEtemp16+500)))}  	: Basal level
	else if (t >= NEtemp17 && t <= NEtemp17+500) {NEn = NE_t2}					: 3rd tone EE
		else if (t > NEtemp17+500 && t < NEtemp18) {NEn = 1.0 + (NE_t2-1)*exp(-Beta2*(t-(NEtemp17+500)))}  	: Basal level	
	else if (t >= NEtemp18 && t <= NEtemp18+500) {NEn = NE_t2}					: 4th tone EE	
		else if (t > NEtemp18+500 && t < NEtemp19) {NEn = 1.0 + (NE_t2-1)*exp(-Beta2*(t-(NEtemp18+500)))}  	: Basal level
	else if (t >= NEtemp19 && t <= NEtemp19+500) {NEn = NE_t3}					: 5th tone EE
		else if (t > NEtemp19+500 && t < NEtemp20) {NEn = 1.0 + (NE_t2-1)*exp(-Beta2*(t-(NEtemp19+500)))}  	: Basal level
	else if (t >= NEtemp20 && t <= NEtemp20+500) {NEn = NE_t3}					: 6th tone EE
		else if (t > NEtemp20+500 && t < NEtemp21) {NEn = 1.0 + (NE_t2-1)*exp(-Beta2*(t-(NEtemp20+500)))}  	: Basal level
	else if (t >= NEtemp21 && t <= NEtemp21+500) {NEn = NE_t3}					: 7th tone EE
		else if (t > NEtemp21+500 && t < NEtemp22) {NEn = 1.0 + (NE_t2-1)*exp(-Beta2*(t-(NEtemp21+500)))}  	: Basal level	
	else if (t >= NEtemp22 && t <= NEtemp22+500) {NEn = NE_t3}					: 8th tone EE	
		else if (t > NEtemp22+500 && t < NEtemp23) {NEn = 1.0 + (NE_t2-1)*exp(-Beta2*(t-(NEtemp22+500)))}  	: Basal level
	else if (t >= NEtemp23 && t <= NEtemp23+500) {NEn = NE_t3}					: 9th tone EE
		else if (t > NEtemp23+500 && t < NEtemp24) {NEn = 1.0 + (NE_t2-1)*exp(-Beta2*(t-(NEtemp23+500)))}  	: Basal level
	else if (t >= NEtemp24 && t <= NEtemp24+500) {NEn = NE_t3}					: 10th tone EE
		else if (t > NEtemp24+500 && t < NEtemp25) {NEn = 1.0 + (NE_t2-1)*exp(-Beta2*(t-(NEtemp24+500)))}  	: Basal level
	else if (t >= NEtemp25 && t <= NEtemp25+500) {NEn = NE_t3}					: 11th tone EE
		else if (t > NEtemp25+500 && t < NEtemp26) {NEn = 1.0 + (NE_t2-1)*exp(-Beta2*(t-(NEtemp25+500)))}  	: Basal level	
	else if (t >= NEtemp26 && t <= NEtemp26+500) {NEn = NE_t3}					: 12th tone EE	
		else if (t > NEtemp26+500 && t < NEtemp27) {NEn = 1.0 + (NE_t2-1)*exp(-Beta2*(t-(NEtemp26+500)))}  	: Basal level
	else if (t >= NEtemp27 && t <= NEtemp27+500) {NEn = NE_t3}					: 13th tone EE
		else if (t > NEtemp27+500 && t < NEtemp28) {NEn = 1.0 + (NE_t2-1)*exp(-Beta2*(t-(NEtemp27+500)))}  	: Basal level
	else if (t >= NEtemp28 && t <= NEtemp28+500) {NEn = NE_t3}					: 14th tone EE
		else if (t > NEtemp28+500 && t < NEtemp29) {NEn = 1.0 + (NE_t2-1)*exp(-Beta2*(t-(NEtemp28+500)))}  	: Basal level
	else if (t >= NEtemp29 && t <= NEtemp29+500) {NEn = NE_t3}					: 15th tone EE
		else if (t > NEtemp29+500 && t < NEtemp30) {NEn = 1.0 + (NE_t2-1)*exp(-Beta2*(t-(NEtemp29+500)))}  	: Basal level	
	else if (t >= NEtemp30 && t <= NEtemp30+500) {NEn = NE_t3}					: 16th tone EE	
		else if (t > NEtemp30+500 && t < NEtemp31) {NEn = 1.0 + (NE_t2-1)*exp(-Beta2*(t-(NEtemp30+500)))}  	: Basal level
	else if (t >= NEtemp31 && t <= NEtemp31+500) {NEn = NE_t3}					: 17th tone EE
		else if (t > NEtemp31+500 && t < NEtemp32) {NEn = 1.0 + (NE_t2-1)*exp(-Beta2*(t-(NEtemp31+500)))}  	: Basal level
	else if (t >= NEtemp32 && t <= NEtemp32+500) {NEn = NE_t3}					: 18th tone EE
		else if (t > NEtemp32+500 && t < NEtemp33) {NEn = 1.0 + (NE_t2-1)*exp(-Beta2*(t-(NEtemp32+500)))}  	: Basal level
	else if (t >= NEtemp33 && t <= NEtemp33+500) {NEn = NE_t3}					: 19th tone EE
		else if (t > NEtemp33+500 && t < NEtemp34) {NEn = 1.0 + (NE_t2-1)*exp(-Beta2*(t-(NEtemp33+500)))}  	: Basal level	
	else if (t >= NEtemp34 && t <= NEtemp34+500) {NEn = NE_t3}					: 20th tone EE		
		else  {	NEn = 1.0}

}
FUNCTION NE2(NEstart2 (ms), NEstop2 (ms)) {
















	LOCAL NE2temp1, NE2temp2, NE2temp3, NE2temp4, NE2temp5, NE2temp6, NE2temp7, NE2temp8, NE2temp9, NE2temp10, NE2temp11, NE2temp12, NE2temp13, NE2temp14, NE2temp15, NE2temp16,s
	NE2temp1 = NEstart2 + 4000
	NE2temp2 = NE2temp1 + 4000
	NE2temp3 = NE2temp2 + 4000
	NE2temp4 = NE2temp3 + 4000
	NE2temp5 = NE2temp4 + 4000
	NE2temp6 = NE2temp5 + 4000
	NE2temp7 = NE2temp6 + 4000
	NE2temp8 = NE2temp7 + 4000
	NE2temp9 = NE2temp8 + 4000
	NE2temp10 = NE2temp9 + 4000
	NE2temp11 = NE2temp10 + 4000
	NE2temp12 = NE2temp11 + 4000 
	NE2temp13 = NE2temp12 + 4000
	NE2temp14 = NE2temp13 + 4000
	NE2temp15 = NE2temp14 + 4000
	
	if (t <= NEstart2) { NE2 = 1.0}
	else if (t >= NEstart2 && t <= NEstop2) {NE2 = NE_S }					: 1st shock
		else if (t > NEstop2 && t < NE2temp1) {NE2 = 1.0 + (NE_S-1)*exp(-Beta2*(t-(NEstop2+500)))} 
	else if (t >= NE2temp1 && t <= NE2temp1+100) {NE2=NE_S}					: 2nd shock
		else if (t > NE2temp1+100 && t < NE2temp2) {NE2 = 1.0 + (NE_S-1)*exp(-Beta2*(t-(NE2temp1+100)))}   				 
	else if (t >= NE2temp2 && t <= NE2temp2+100) {NE2=NE_S}					: 3rd shock
		else if (t > NE2temp2+100 && t < NE2temp3) {NE2 = 1.0 + (NE_S-1)*exp(-Beta2*(t-(NE2temp2+100)))}  				 
	else if (t >= NE2temp3 && t <= NE2temp3+100) {NE2=NE_S}					: 4th shock
		else if (t > NE2temp3+100 && t < NE2temp4) {NE2 = 1.0 + (NE_S-1)*exp(-Beta2*(t-(NE2temp3+100)))}  				 
	else if (t >= NE2temp4 && t <= NE2temp4+100) {NE2=NE_S}					: 5th shock
		else if (t > NE2temp4+100 && t < NE2temp5) {NE2 = 1.0 + (NE_S-1)*exp(-Beta2*(t-(NE2temp4+100)))}  				 
	else if (t >= NE2temp5 && t <= NE2temp5+100) {NE2=NE_S}					: 6th shock
		else if (t > NE2temp5+100 && t < NE2temp6) {NE2 = 1.0 + (NE_S-1)*exp(-Beta2*(t-(NE2temp5+100)))} 				 
	else if (t >= NE2temp6 && t <= NE2temp6+100) {NE2=NE_S}					: 7th shock
		else if (t > NE2temp6+100 && t < NE2temp7) {NE2 = 1.0 + (NE_S-1)*exp(-Beta2*(t-(NE2temp6+100)))}  				 
	else if (t >= NE2temp7 && t <= NE2temp7+100) {NE2=NE_S}					: 8th shock
		else if (t > NE2temp7+100 && t < NE2temp8) {NE2 = 1.0 + (NE_S-1)*exp(-Beta2*(t-(NE2temp7+100)))}  				    
	else if (t >= NE2temp8 && t <= NE2temp8+100) {NE2=NE_S }					: 9th shock
		else if (t > NE2temp8+100 && t < NE2temp9) {NE2 = 1.0 + (NE_S-1)*exp(-Beta2*(t-(NE2temp8+100)))}  				    
	else if (t >= NE2temp9 && t <= NE2temp9+100) {NE2=NE_S }					: 10th shock
		else if (t > NE2temp9+100 && t < NE2temp10) {NE2 = 1.0 + (NE_S-1)*exp(-Beta2*(t-(NE2temp9+100)))}  				    
	else if (t >= NE2temp10 && t <= NE2temp10+100) {NE2=NE_S}					: 11th shock
		else if (t > NE2temp10+100 && t < NE2temp11) {NE2 = 1.0 + (NE_S-1)*exp(-Beta2*(t-(NE2temp10+100)))}  				 
	else if (t >= NE2temp11 && t <= NE2temp11+100) {NE2=NE_S }					: 12th shock
		else if (t > NE2temp11+100 && t < NE2temp12) {NE2 = 1.0 + (NE_S-1)*exp(-Beta2*(t-(NE2temp11+100)))}  				 
	else if (t >= NE2temp12 && t <= NE2temp12+100) {NE2=NE_S}					: 13th shock
		else if (t > NE2temp12+100 && t < NE2temp13) {NE2 = 1.0 + (NE_S-1)*exp(-Beta2*(t-(NE2temp12+100)))} 				 
	else if (t >= NE2temp13 && t <= NE2temp13+100) {NE2=NE_S }					: 14th shock
		else if (t > NE2temp13+100 && t < NE2temp14) {NE2 = 1.0 + (NE_S-1)*exp(-Beta2*(t-(NE2temp13+100)))}   				 
	else if (t >= NE2temp14 && t <= NE2temp14+100) {NE2=NE_S}					: 15th shock
		else if (t > NE2temp14+100 && t < NE2temp15) {NE2 = 1.0 + (NE_S-1)*exp(-Beta2*(t-(NE2temp14+100)))}  				 
	else if (t >= NE2temp15 && t <= NE2temp15+100) {NE2=NE_S}					: 16th shock
		else  {	NE2 = 1.0}
}
FUNCTION GAP1(GAPstart1 (ms), GAPstop1 (ms)) {
	LOCAL s
	if (t <= GAPstart1) { GAP1 = 1}
	else if (t >= GAPstop1 ) {GAP1 = 1}					: During the Gap, apply lamda2*2
	else  {	GAP1 = 1}
}
FUNCTION unirand() {    : uniform random numbers between 0 and 1
        unirand = scop_random()
}
