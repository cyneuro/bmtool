:Pyramidal Cells to Interneuron Cells AMPA+NMDA with local Ca2+ pool

NEURON {
	POINT_PROCESS pyrD2interD_STFD
	USEION ca READ eca	
	NONSPECIFIC_CURRENT inmda, iampa
	RANGE initW
	RANGE Cdur_nmda, AlphaTmax_nmda, Beta_nmda, Erev_nmda, gbar_nmda, W_nmda, on_nmda, g_nmda
	RANGE Cdur_ampa, AlphaTmax_ampa, Beta_ampa, Erev_ampa, gbar_ampa, W, on_ampa, g_ampa
	RANGE eca, ICan, P0n, fCan, tauCa, Icatotal
	RANGE ICaa, P0a, fCaa
	RANGE Cainf, pooldiam, z
	RANGE lambda1, lambda2, threshold1, threshold2
	RANGE fmax, fmin, Wmax, Wmin, maxChange, normW, scaleW, srcid, destid,limitW
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

	Cdur_ampa = 0.713 (ms)
	AlphaTmax_ampa = 10.1571 (/ms)
	Beta_ampa = 0.4167 (/ms)
	Erev_ampa = 0 (mV)
	gbar_ampa = 1e-3 (uS)

	eca = 120

	Cainf = 50e-6 (mM)
	pooldiam =  1.8172 (micrometer)
	z = 2
	neuroM = 0

	tauCa = 50 (ms)
	P0n = .015
	fCan = .024
	
	P0a = .001
	fCaa = .024
	
	lambda1 = 8 : 3 : 10 :6 : 4 :2
	lambda2 = .01
	threshold1 = 0.35 : 0.4 :  0.45 :0.5 (uM)
	threshold2 = 0.4 : 0.45 :  0.5 :0.6 (uM)

	:AMPA Weight
	initW = 1.5 : 1.5 : 2 : 0.1:3 : 2 :3
	fmax = 4 : 8 : 5: 4 :3
	fmin = .8	

	thr_rp = 1 : .7
	
	facfactor = 1
	: the (1) is needed for the range limits to be effective
        f = 1 (1) < 0, 1e9 >    : facilitation  : 1.3 (1) < 0, 1e9 >    : facilitation
        tauF = 45 (ms) < 1e-9, 1e9 >
        d1 = 0.95 (1) < 0, 1 >: 0.95 (1) < 0, 1 >     : fast depression
        tauD1 = 40 (ms) < 1e-9, 1e9 >
        d2 = 0.9 (1) < 0, 1 > : 0.9 (1) < 0, 1 >     : slow depression
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

	ICan (nA)
	ICaa (nA)
	Afactor	(mM/ms/nA)
	Icatotal (nA)

	dW_ampa
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

STATE { r_nmda r_ampa capoolcon W}

INITIAL {
	on_nmda = 0
	r_nmda = 0
	W_nmda = initW

	on_ampa = 0
	r_ampa = 0
	W = initW
	limitW = 1

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
	: if (W > Wmax) { 
	:	W = Wmax
	: } else if (W < Wmin) {
 		: W = Wmin
	: }
	
	 if (neuroM==0) {
	g_nmda = gbar_nmda*r_nmda*facfactor
	} else {
	g_nmda = gbar_nmda*r_nmda*facfactor
	}
	inmda = W_nmda*g_nmda*(v - Erev_nmda)*sfunc(v)

	g_ampa = gbar_ampa*r_ampa*facfactor
	iampa = W*g_ampa*(v - Erev_ampa)

	ICan = P0n*g_nmda*(v - eca)*sfunc(v)
	ICaa = P0a*W*g_ampa*(v-eca)/initW	
	Icatotal = ICan + ICaa
}

DERIVATIVE release {
    : W' = eta(capoolcon)*(lambda1*omega(capoolcon, threshold1, threshold2)-lambda2*W)	  : Long-term plasticity was implemented. (Shouval et al. 2002a, 2002b)
   
	    W' = 1e-12*limitW*eta(capoolcon)*(lambda1*omega(capoolcon, threshold1, threshold2)-lambda2*W)	  : Long-term plasticity was implemented. (Shouval et al. 2002a, 2002b)

	r_nmda' = AlphaTmax_nmda*on_nmda*(1-r_nmda)-Beta_nmda*r_nmda
	r_ampa' = AlphaTmax_ampa*on_ampa*(1-r_ampa)-Beta_ampa*r_ampa
  
	capoolcon'= -fCan*Afactor*Icatotal + (Cainf-capoolcon)/tauCa
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
	
	if (flag == 0) {  : Short term plasticity was implemented(Varela et. al 1997):
	
	rp = unirand()	
	
	:F  = 1 + (F-1)* exp(-(t - tsyn)/tauF)
	D1 = 1 - (1-D1)*exp(-(t - tsyn)/tauD1)
	D2 = 1 - (1-D2)*exp(-(t - tsyn)/tauD2)
 :printf("%g\t%g\t%g\t%g\t%g\t%g\n", t, t-tsyn, F, D1, D2, facfactor)
	::printf("%g\t%g\t%g\t%g\n", F, D1, D2, facfactor)
	tsyn = t
	
	facfactor = F * D1 * D2

	F = F * f
	
	if (F > 2) { 
	F=2
	}
	if (facfactor < 0.7) { 
	facfactor=0.7
	}
	if (F < 0.8) { 
	F=0.8
	}	
	D1 = D1 * d1
	D2 = D2 * d2
:printf("\t%g\t%g\t%g\n", F, D1, D2)
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
FUNCTION unirand() {    : uniform random numbers between 0 and 1
        unirand = scop_random()
}