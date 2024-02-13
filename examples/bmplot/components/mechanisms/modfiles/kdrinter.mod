: potassium delayed rectifier channel (interneuron)

NEURON {
	SUFFIX kdrinter
	USEION k READ ek WRITE ik
	RANGE gkdrbar, ik, gkdr, inf, tau
}

UNITS {
	(mA) = (milliamp)
	(mV) = (millivolt)
}

PARAMETER {
	v (mV)
	dt (ms)
	gkdrbar = 0.008 (mho/cm2) <0,1e9>
	ek = -80 (mV)
}

STATE {
	n
}

ASSIGNED {
	ik (mA/cm2)
	inf
	tau (ms)
	gkdr (mho/cm2)
}

INITIAL {
	rate(v)
	n = inf
}

BREAKPOINT {
	SOLVE states METHOD cnexp
	gkdr = gkdrbar*n*n*n*n
	ik = gkdr*(v-ek)
}

DERIVATIVE states {
	rate(v)
	n' = (inf-n)/tau
}

UNITSOFF

FUNCTION alf(v(mV)) { 
	alf = 0.15*exp((v+19)/10.67)
}

FUNCTION bet(v(mV)) {
	bet = 0.15*exp(-(v+19)/42.68)
}	

PROCEDURE rate(v(mV)) { LOCAL sum, nalf, nbet
	
	nalf = alf(v)
	nbet = bet(v) 
	
	sum = nalf+nbet
	inf = nalf/sum
	tau = 1/(sum)
}

UNITSON	
