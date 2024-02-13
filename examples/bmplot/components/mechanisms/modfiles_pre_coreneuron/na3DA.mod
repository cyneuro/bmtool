TITLE na3
: Na current 
: from Jeff M.
:  ---------- modified -------M.Migliore may97

NEURON {
	SUFFIX na3DA
	USEION na READ ena WRITE ina
	RANGE  gbar, ar2
	GLOBAL minf, hinf, mtau, htau, sinf, taus,qinf, thinf
}

PARAMETER {
	tone_period = 4000   
	DA_period = 500
	DA_start = 64000		    : D1R(Low Affinity) Dopamine Effect after 6 conditioning trials (14*4000) = 64000)
	DA_stop = 96000
	DA_ext1 = 196000
	DA_ext2 = 212000
	DA_t1 = -0.1 : -0.3 : -0.15            : Amount of DA effect- negative value decreases AP threshold / positive value increases threshold of AP

	DA_period2 = 100
	DA_start2 = 36000		   : shock Dopamine Effect during shock after 1 conditioning trial
	DA_t2 = -1 : -1 : -0.1           : Amount(%) of DA effect- negative value decreases AP threshold / positive value increases threshold of AP	
	
	gbar = 0.010   	(mho/cm2)	
								
	tha  =  -30	(mV)		: v 1/2 for act	
	qa   = 7.2	(mV)		: act slope (4.5)		
	Ra   = 0.4	(/ms)		: open (v)		
	Rb   = 0.124 	(/ms)		: close (v)		

	thi1  = -45	(mV)		: v 1/2 for inact 	
	thi2  = -45 	(mV)		: v 1/2 for inact 	
	qd   = 1.5	(mV)	        : inact tau slope
	qg   = 1.5      (mV)
	mmin=0.02	
	hmin=0.5			
	q10=2
	Rg   = 0.01 	(/ms)		: inact recov (v) 	
	Rd   = .03 	(/ms)		: inact (v)	
	qq   = 10        (mV)
	tq   = -55      (mV)

	thinf  = -50 	(mV)		: inact inf slope	
	qinf  = 4 	(mV)		: inact inf slope 

        vhalfs=-60	(mV)		: slow inact.
        a0s=0.0003	(ms)		: a0s=b0s
        zetas=12	(1)
        gms=0.2		(1)
        smax=10		(ms)
        vvh=-58		(mV) 
        vvs=2		(mV)
        ar2=1		(1)		: 1=no inact., 0=max inact.
	ena		(mV)            : must be explicitly def. in hoc
	celsius
	v 		(mV)
}


UNITS {
	(mA) = (milliamp)
	(mV) = (millivolt)
	(pS) = (picosiemens)
	(um) = (micron)
} 

ASSIGNED {
	ina 		(mA/cm2)
	thegna		(mho/cm2)
	minf 		hinf 		
	mtau (ms)	htau (ms) 	
	sinf (ms)	taus (ms)
	tha1	
}
 

STATE { m h s}

BREAKPOINT {
        SOLVE states METHOD cnexp
        thegna = gbar*m*m*m*h*s
	ina = thegna * (v - ena)
} 

INITIAL {
	trates(v,ar2)
	m=minf  
	h=hinf
	s=sinf
}


FUNCTION alpv(v(mV)) {
         alpv = 1/(1+exp((v-vvh)/vvs))
}
        
FUNCTION alps(v(mV)) {  
  alps = exp(1.e-3*zetas*(v-vhalfs)*9.648e4/(8.315*(273.16+celsius)))
}

FUNCTION bets(v(mV)) {
  bets = exp(1.e-3*zetas*gms*(v-vhalfs)*9.648e4/(8.315*(273.16+celsius)))
}

LOCAL mexp, hexp, sexp

DERIVATIVE states {   
        trates(v,ar2)      
        m' = (minf-m)/mtau
        h' = (hinf-h)/htau
        s' = (sinf - s)/taus
}

PROCEDURE trates(vm,a2) {  
        LOCAL  a, b, c, qt
        qt=q10^((celsius-24)/10)
		tha1 = tha + DA1(t)	+ DA2(t)
	a = trap0(vm,tha1,Ra,qa)
	b = trap0(-vm,-tha1,Rb,qa)
	mtau = 1/(a+b)/qt
        if (mtau<mmin) {mtau=mmin}
		
	if (v < -57.5 ) {
	minf = 0
	} else{
	minf = a/(a+b)
	}
	
	a = trap0(vm,thi1,Rd,qd)
	b = trap0(-vm,-thi2,Rg,qg)
	htau =  1/(a+b)/qt
        if (htau<hmin) {htau=hmin}
	hinf = 1/(1+exp((vm-thinf)/qinf))
	c=alpv(vm)
        sinf = c+a2*(1-c)
        taus = bets(vm)/(a0s*(1+alps(vm)))
        if (taus<smax) {taus=smax}
}

FUNCTION trap0(v,th,a,q) {
	if (fabs(v-th) > 1e-6) {
	        trap0 = a * (v - th) / (1 - exp(-(v - th)/q))
	} else {
	        trap0 = a * q
 	}
}	
FUNCTION DA1(t) {
	    if (t >= DA_start && t <= DA_stop){ 									: During conditioning
			if ((t/tone_period-floor(t/tone_period)) >= (1-DA_period/tone_period)) {DA1 = DA_t1}
			else if ((t/tone_period-floor(t/tone_period)) == 0) {DA1 = DA_t1}
			else {DA1 = 0}}
		else if (t >= DA_ext1 && t <= DA_ext2){								: During 4trials of Extinction
			if ((t/tone_period-floor(t/tone_period)) >= (1-DA_period/tone_period)) {DA1 = DA_t1}
			else if ((t/tone_period-floor(t/tone_period)) == 0) {DA1 = DA_t1}
			else {DA1 = 0}}		
		else  {DA1 = 0}
	}
FUNCTION DA2(t) {
	    if (t >= DA_start2 && t <= DA_stop){
			if((t/tone_period-floor(t/tone_period)) >= (1-DA_period2/tone_period)) {DA2 = DA_t2}
			else if ((t/tone_period-floor(t/tone_period)) == 0) {DA2 = DA_t2}
			else  {DA2 = 0}}
		else  {DA2 = 0}
	}