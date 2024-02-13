TITLE K-DR channel
: from Klee Ficker and Heinemann
: modified to account for Dax et al.
: M.Migliore 1997

UNITS {
	(mA) = (milliamp)
	(mV) = (millivolt)

}

PARAMETER {

	tone_period = 4000    
	DA_period = 500	
	DA_start = 64000		             : D1R(Low Affinity) Dopamine Effect after 6 conditioning trials (15*4000) = 60000)
	DA_stop = 96000
	DA_ext1 = 196000
	DA_ext2 = 212000	
	
	DA_t1 = 0.95 : 0.9 : 1 :  1 : 0.9           : Amount(%) of DA effect- negative value decreases AP threshold / positive value increases threshold of AP
	DA_period2 = 100
	DA_start2 = 36000		   			: shock Dopamine Effect during shock after 1 conditioning trial
	DA_t2 = .8           				: Amount(%) of DA effect- negative value decreases AP threshold / positive value increases threshold of AP	

	v (mV)
        ek (mV)		: must be explicitely def. in hoc
	celsius		(degC)
	gbar=.003 (mho/cm2)
        vhalfn = -15: 13 : -25  : -20  (mV)
        a0n=0.02      (/ms)
        zetan=-3    (1)
        gmn=0.7  (1)
	nmax=2  (1)
	qt=1
}


NEURON {
	SUFFIX kdrDA
	USEION k READ ek WRITE ik
        RANGE gkdr, i, gbar
	RANGE ninf,taun
}

STATE {
	n
}

ASSIGNED {
	ik (mA/cm2)
	i  (mA/cm2)
        ninf
        gkdr
        taun
}

BREAKPOINT {
	SOLVE states METHOD cnexp
	gkdr = gbar*n
	ik = gkdr*(v-ek)*DA1(t)*DA2(t)
	i = ik

}

INITIAL {
	rates(v)
	n=ninf
}


FUNCTION alpn(v(mV)) {
  alpn = exp(1.e-3*(-3)*(v-vhalfn)*9.648e4/(8.315*(273.16+celsius))) 
}

FUNCTION betn(v(mV)) {
  betn = exp(1.e-3*(-3)*(0.7)*(v-vhalfn)*9.648e4/(8.315*(273.16+celsius))) 
}

DERIVATIVE states {     : exact when v held constant; integrates over dt step
        rates(v)
        n' = (ninf - n)/taun
}

PROCEDURE rates(v (mV)) { :callable from hoc
        LOCAL a
        a = alpn(v)
		if (v < -55 ) {              ::::::::::::::::::::   -55
		ninf = 0
		} else{
		ninf = 1 / ( 1 + exp( ( vhalfn - v ) / 11 ) )
		:ninf = 1 / ( 1 + exp( ( - v + 13 ) / 8.738 ) )
        }
		taun = betn(v)/(qt*(0.02)*(1+a))
	if (taun<nmax) {taun=nmax}
}


FUNCTION DA1(t) {
	    if (t >= DA_start && t <= DA_stop){ 									: During conditioning
			if ((t/tone_period-floor(t/tone_period)) >= (1-DA_period/tone_period)) {DA1 = DA_t1}
			else if ((t/tone_period-floor(t/tone_period)) == 0) {DA1 = DA_t1}
			else {DA1 = 1}}
		else if (t >= DA_ext1 && t <= DA_ext2){								: During 4trials of Extinction
			if ((t/tone_period-floor(t/tone_period)) >= (1-DA_period/tone_period)) {DA1 = DA_t1}
			else if ((t/tone_period-floor(t/tone_period)) == 0) {DA1 = DA_t1}
			else {DA1 = 1}}		
		else  {DA1 = 1}
	}
FUNCTION DA2(t) {
	    if (t >= DA_start2 && t <= DA_stop){
			if((t/tone_period-floor(t/tone_period)) >= (1-DA_period2/tone_period)) {DA2 = DA_t2}
			else if ((t/tone_period-floor(t/tone_period)) == 0) {DA2 = DA_t2}
			else  {DA2 = 1}}
		else  {DA2 = 1}
	}