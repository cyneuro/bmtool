:Pyramidal Cells to Pyramidal Cells AMPA+NMDA with local Ca2+ pool

NEURON {
	POINT_PROCESS pyrD2pyrD_STFD
	USEION ca READ eca	
	NONSPECIFIC_CURRENT inmda, iampa
	RANGE initW
	RANGE Cdur_nmda, AlphaTmax_nmda, Beta_nmda, Erev_nmda, gbar_nmda, W_nmda, on_nmda, g_nmda
	RANGE Cdur_ampa, AlphaTmax_ampa, Beta_ampa, Erev_ampa, gbar_ampa, W, on_ampa, g_ampa
	RANGE eca, ICa, P0, fCa, tauCa, iCatotal
	RANGE Cainf, pooldiam, z
	RANGE lambda1, lambda2, threshold1, threshold2
	RANGE fmax, fmin, Wmax, Wmin, maxChange, normW, scaleW, limitW, srcid,destid,tempW 
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
	
	Cdur_nmda = 16.7650 (ms)
	AlphaTmax_nmda = .2659 (/ms)
	Beta_nmda = 0.008 (/ms)
	Erev_nmda = 0 (mV)
	gbar_nmda = .5e-3 (uS)

	Cdur_ampa = 1.4210 (ms)
	AlphaTmax_ampa = 3.8142 (/ms)
	Beta_ampa =  0.1429(/ms) :0.1429 as original 0.2858 as half,0.07145 as twice
	Erev_ampa = 0 (mV)
	gbar_ampa = 1e-3 (uS)

	eca = 120

	Cainf = 50e-6 (mM)
	pooldiam =  1.8172 (micrometer)
	z = 2
	neuroM = 0
	tauCa = 50 (ms)
	P0 = .015
	fCa = .024
	
	lambda1 = 40 : 60 : 12 :80: 20 : 15 :8 :5: 2.5
	lambda2 = .03
	threshold1 = 0.4 :  0.45 : 0.35 :0.35:0.2 :0.50 (uM)
	threshold2 = 0.55 : 0.50 : 0.40 :0.4 :0.3 :0.60 (uM)

	initW = 5.0 : 1.0 :  0.9 : 0.8 : 2 : 10 : 6 :1.5
	fmax = 3 : 2.5 : 4 : 2 : 3 : 1.5 : 3
	fmin = .8
	
	DAstart1 = 39500
	DAstop1 = 40000	
	DAstart2 = 35900
	DAstop2 = 36000	

	DA_t1 = 1.2
	DA_t2 = 0.8 : 0.9
    DA_t3 = 0.9
	DA_S = 1.3 : 0.95 : 0.6	
	Beta1 = 0.001  (/ms) : 1/decay time for neuromodulators
	Beta2 = 0.0001  (/ms)

	thr_rp = 1 : .7
	
	facfactor = 1
	: the (1) is needed for the range limits to be effective
        f = 0 (1) < 0, 1e9 >    : facilitation
        tauF = 20 (ms) < 1e-9, 1e9 >
        d1 = 0.95 (1) < 0, 1 >     : fast depression
        tauD1 = 40 (ms) < 1e-9, 1e9 >
        d2 = 0.9 (1) < 0, 1 >     : slow depression
        tauD2 = 70 (ms) < 1e-9, 1e9 >		
}

ASSIGNED {
	v (mV)

	inmda (nA)
	g_nmda (uS)
	on_nmda
	W_nmda

	iampa (nA)
	g_ampa (uS)
	on_ampa
	: W
	limitW

	t0 (ms)

	ICa (mA)
	Afactor	(mM/ms/nA)
	iCatotal (mA)

	dW_ampa
	Wmax
	Wmin
	maxChange
	normW
	scaleW
	
    tempW
	pregid
	postgid

	rp
	tsyn
	
	fa
	F
	D1
	D2
}

STATE { r_nmda r_ampa capoolcon W}

INITIAL {
	on_nmda = 0
	r_nmda = 0
	W_nmda = initW

	on_ampa = 0
	r_ampa = 0
	W = initW
    limitW = 1
    
	tempW = initW
	t0 = -1

	Wmax = fmax*initW
	Wmin = fmin*initW
	maxChange = (Wmax-Wmin)/10
	dW_ampa = 0

	capoolcon = Cainf
	Afactor	= 1/(z*FARADAY*4/3*pi*(pooldiam/2)^3)*(1e6)

	fa =0
	F = 1
	D1 = 1
	D2 = 1
}

BREAKPOINT {

if ((eta(capoolcon)*(lambda1*omega(capoolcon, threshold1, threshold2)-lambda2*W))>0&&W>=Wmax) {
        limitW=1e-12
	} else if ((eta(capoolcon)*(lambda1*omega(capoolcon, threshold1, threshold2)-lambda2*W))<0&&W<=Wmin) {
        limitW=1e-12
	} else {
	limitW=1 }
	
	SOLVE release METHOD cnexp
	if (t0>0) {
		if (rp < thr_rp) {
			if (t-t0 < Cdur_ampa) {
				on_ampa = 1
			} else {
				on_ampa = 0
			}
		} else {
			on_ampa = 0
		}
	}
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
	g_nmda = gbar_nmda*r_nmda*facfactor*DA1(DAstart1,DAstop1)*DA2(DAstart2,DAstop2)        : Dopamine effect on NMDA to reduce NMDA current amplitude
		} else {
		g_nmda = gbar_nmda*r_nmda*facfactor
		}
		inmda = W_nmda*g_nmda*(v - Erev_nmda)*sfunc(v)

	g_ampa = gbar_ampa*r_ampa*facfactor
	iampa = W*g_ampa*(v - Erev_ampa)

	ICa = P0*g_nmda*(v - eca)*sfunc(v)
	
}

DERIVATIVE release {
	: W' = eta(capoolcon)*(lambda1*omega(capoolcon, threshold1, threshold2)-lambda2*W)	  : Long-term plasticity was implemented. (Shouval et al. 2002a, 2002b)
	
	W' = 1e-12*limitW*eta(capoolcon)*(lambda1*omega(capoolcon, threshold1, threshold2)-lambda2*W)	  : Long-term plasticity was implemented. (Shouval et al. 2002a, 2002b)
	r_nmda' = AlphaTmax_nmda*on_nmda*(1-r_nmda)-Beta_nmda*r_nmda
	r_ampa' = AlphaTmax_ampa*on_ampa*(1-r_ampa)-Beta_ampa*r_ampa
  	capoolcon'= -fCa*Afactor*ICa + (Cainf-capoolcon)/tauCa
}

NET_RECEIVE(dummy_weight) {
	    if (flag==0) {           :a spike arrived, start onset state if not already on
         if ((!on_nmda)){       :this synpase joins the set of synapses in onset state
           t0=t
	      on_nmda=1		
	      net_send(Cdur_nmda,1)  
         } else if (on_nmda==1) {             :already in onset state, so move offset time
          net_move(t+Cdur_nmda)
		  t0=t
	      }
         }		  
	if (flag == 1) { : turn off transmitter, i.e. this synapse enters the offset state	
	on_nmda=0
    }
	         
	if (flag == 0) {   : Short term plasticity was implemented(Varela et. al 1997):
	D1 = 1 - (1-D1)*exp(-(t - tsyn)/tauD1)
	D2 = 1 - (1-D2)*exp(-(t - tsyn)/tauD2)
	tsyn = t
	
	facfactor = F * D1 * D2	
	if (F > 3) { 
	F=3	}
	if (facfactor < 0.5) { 
	facfactor=0.5
	}	
	D1 = D1 * d1
	D2 = D2 * d2
	}
}

:::::::::::: FUNCTIONs and PROCEDUREs ::::::::::::

FUNCTION sfunc (v (mV)) {
	UNITSOFF
	sfunc = 1/(1+0.33*exp(-0.06*v))
	UNITSON
}

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
FUNCTION unirand() {    : uniform random numbers between 0 and 1
        unirand = scop_random()
}
