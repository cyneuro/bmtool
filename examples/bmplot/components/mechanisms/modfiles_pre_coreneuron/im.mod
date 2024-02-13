: voltage-gated persistent muscarinic channel

NEURON {
	SUFFIX im
	USEION k READ ek WRITE ik
	RANGE gm, i,  gbar
	RANGE ninf, taun
}

UNITS {
	(mA) = (milliamp)
	(mV) = (millivolt)
}

PARAMETER {
	gbar = 0.0003 (siemens/cm2) <0,1e9>
}

ASSIGNED {
	v (mV)
	ek (mV)
	ik (mA/cm2)
	i  (mA/cm2)
	ninf
	taun (ms)
	gm (siemens/cm2)
}

STATE {
	n
}

BREAKPOINT {
	SOLVE states METHOD cnexp
	gm = gbar*n*n
	ik = gm*(v-ek)
	i = ik
}

INITIAL {
	rate(v)
	n = ninf
}

DERIVATIVE states {
	rate(v)
	n' = (ninf-n)/taun
}

FUNCTION alf(v (mV)) (/ms) {
	UNITSOFF
	alf = 0.016/exp(-(v+52.7)/23)
	UNITSON
}

FUNCTION bet(v (mV)) (/ms) {
	UNITSOFF
	bet = 0.016/exp((v+52.7)/18.8)
	UNITSON
}

PROCEDURE rate(v (mV)) {
	LOCAL sum, aa, ab
	UNITSOFF
	aa=alf(v) ab=bet(v) 
	
	sum = aa+ab
	if (v < -67.5 ) {
	ninf = 0
	} else {
	ninf = 1 / ( 1 + exp( ( - v - 52.7 ) / 10.34 ) )
	}
	taun = 1/sum
	UNITSON
}
