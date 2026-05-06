TITLE K-DR channel
: from Klee Ficker and Heinemann
: modified to account for Dax et al.
: M.Migliore 1997

UNITS {
	(mA) = (milliamp)
	(mV) = (millivolt)

}

PARAMETER {

	v (mV)
        ek (mV)		: must be explicitely def. in hoc
	celsius		(degC)
	gbar=.003 (mho/cm2)
        vhalfn = 0 :-15: 13 : -25  : -20  (mV)
        a0n=0.02      (/ms)
        zetan=-3    (1)
        gmn=0.7  (1)
	nmax=2  (1)
	qt=1
}


NEURON {
	SUFFIX kdrCA3
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
	ik = gkdr*(v-ek)
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
		if (v < -58 ) {              ::::::::::::::::::::   -55
		ninf = 0
		} else{
		ninf = 1 / ( 1 + exp( ( vhalfn - v ) / 11 ) ) :/11
		:ninf = 1 / ( 1 + exp( ( - v + 13 ) / 8.738 ) )
        }
		taun = betn(v)/(qt*(0.08)*(1+a))
	if (taun<nmax) {taun=nmax}
}
		