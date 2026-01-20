COMMENT
Wang-Buzsaki Model of an Inhibitory
Interneuron in Rat Hippocampus

Potassium Channel

Reference: Borgers - An Introduction to Modeling Neuronal Dynamics Chapter 5
.mod by Tyler Banks
ENDCOMMENT

NEURON {
	SUFFIX k_wb
	USEION k READ ek WRITE ik
	RANGE gbar, g
	RANGE inf, tau
	RANGE ik
    RANGE ninfvhalf,ninfk,tauphi
}

UNITS {
	(mA) = (milliamp)
	(mV) = (millivolt)
}

PARAMETER {
	gbar (siemens/cm2)
    ninfvhalf = 30.03
    ninfk = -17.37
	tauphi = 5
}

ASSIGNED {
	v (mV)
	ek (mV)
	ik (mA/cm2)
	g (siemens/cm2)
	inf
	tau (ms)
}

STATE {
	n
}

BREAKPOINT {
	SOLVE states METHOD cnexp
	g = gbar*n*n*n*n
	ik = g*(v-ek)
}

INITIAL {
	rate(v)
	n = inf
}

DERIVATIVE states {
	rate(v)
	n' = (inf-n)/tau
}

COMMENT

function n_i_inf=n_i_inf(v)
alpha_n=-0.01*(v+34)./(exp(-0.1*(v+34))-1);
beta_n=0.125*exp(-(v+44)/80);
n_i_inf=alpha_n./(alpha_n+beta_n);
:inf = -0.01*(v+34)/(exp(-0.1*(v+34))-1)/(-0.01*(v+34)/(exp(-0.1*(v+34))-1)+0.125*exp(-(v+44)/80))
:tau = 1/(-0.01*(v+34)/(exp(-0.1*(v+34))-1)+0.125*exp(-(v+44)/80))

Regression fit INF
ninf = 1.0/(1.0+(exp((v+30.03)/(-17.37))))

Calculated TAU
ntau = (exp(0.1*v) - 0.03337)/((0.01*v + 0.34)*exp(0.1*v) + 0.125*(exp(0.1*v) - 0.03337)*exp(-v/80 - 11/20))

ENDCOMMENT

PROCEDURE rate(v (mV)) {
	UNITSOFF
	inf = 1.0/(1.0+(exp((v+ninfvhalf)/(ninfk))))     
	tau = (exp(0.1*v) - 0.03337)/((0.01*v + 0.34)*exp(0.1*v) + 0.125*(exp(0.1*v) - 0.03337)*exp(-v/80 - 11/20))
	:inf = -0.01*(v+34)/(exp(-0.1*(v+34))-1)/(-0.01*(v+34)/(exp(-0.1*(v+34))-1)+0.125*exp(-(v+44)/80))
	:tau = 1/(-0.01*(v+34)/(exp(-0.1*(v+34))-1)+0.125*exp(-(v+44)/80))
	tau = tau/tauphi
	UNITSON
}