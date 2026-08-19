COMMENT
Reduced Traub-Miles Model of a Pyramidal
Neuron in Rat Hippocampus

Potassium Channel

Reference: Borgers - An Introduction to Modeling Neuronal Dynamics Chapter 5
.mod by Tyler Banks
ENDCOMMENT

NEURON {
	SUFFIX k_rtm
	USEION k READ ek WRITE ik
	RANGE gbar, g
	RANGE inf, tau
	RANGE ik
    RANGE ninfvhalf,ninfk
}

UNITS {
	(mA) = (milliamp)
	(mV) = (millivolt)
}

PARAMETER {
	gbar (siemens/cm2)
    ninfvhalf = 40.8
    ninfk = -11.03
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

function n_e_inf=n_e_inf(v)
alpha_n=0.032*(v+52)./(1-exp(-(v+52)/5));
beta_n=0.5*exp(-(v+57)/40);
n_e_inf=alpha_n./(alpha_n+beta_n);
:inf = 0.032*(v+52)/(1-exp(-(v+52)/5))/(0.032*(v+52)/(1-exp(-(v+52)/5))+0.5*exp(-(v+57)/40))
:tau = 1/(0.032*(v+52)/(1-exp(-(v+52)/5))+0.5*exp(-(v+57)/40))

Regression fit INF
ninf = 1.0/(1.0+(exp((v+40.8)/(-11.03))))

Calculated TAU
ntau = (1 - exp(-v/5 - 52/5))/(0.032*v + 0.5*(1 - exp(-v/5 - 52/5))*exp(-v/40 - 57/40) + 1.664)

ENDCOMMENT

PROCEDURE rate(v (mV)) {
	UNITSOFF
	inf = 1.0/(1.0+(exp((v+ninfvhalf)/(ninfk))))   
	:inf = 0.032*(v+52)/(1-exp(-(v+52)/5))/(0.032*(v+52)/(1-exp(-(v+52)/5))+0.5*exp(-(v+57)/40))    
	tau = (1 - exp(-v/5 - 52/5))/(0.032*v + 0.5*(1 - exp(-v/5 - 52/5))*exp(-v/40 - 57/40) + 1.664)
	:tau = 1/(0.032*(v+52)/(1-exp(-(v+52)/5))+0.5*exp(-(v+57)/40))
	UNITSON
}