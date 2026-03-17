COMMENT
Reduced Traub-Miles Model of a Pyramidal
Neuron in Rat Hippocampus

Sodium Channel

Reference: Borgers - An Introduction to Modeling Neuronal Dynamics Chapter 5
.mod by Tyler Banks
ENDCOMMENT

NEURON {
	SUFFIX na_rtm
	USEION na READ ena WRITE ina
	RANGE gbar, g
	RANGE minf, hinf, mtau, htau
	RANGE ina
    RANGE minfvhalf,minfk,hinfvhalf,hinfk
}

UNITS {
	(mA) = (milliamp)
	(mV) = (millivolt)
}

PARAMETER {
	gbar (siemens/cm2)
    minfvhalf = 41.03
    minfk = -7.1 :-7.41
    hinfvhalf = 45.32
    hinfk = 4.04
}

ASSIGNED {
	v (mV)
	ena (mV)
	ina (mA/cm2)
	minf
	hinf
	mtau (ms)
	htau (ms)
	g (siemens/cm2)
}

STATE {
	m h
}

BREAKPOINT {
	SOLVE states METHOD cnexp
	m = minf : See Borgers Page 32 Figure 5.1 for explaination
	g = gbar*m*m*m*h
	ina = g*(v-ena)
}

INITIAL {
	rate(v)
	m = minf
	h = hinf
}

DERIVATIVE states {
	rate(v)
	:m' = (minf-m)/mtau
	h' = (hinf-h)/htau
}

COMMENT

function m_e_inf=m_e_inf(v)
alpha_m=0.32*(v+54)./(1-exp(-(v+54)/4));
beta_m=0.28*(v+27)./(exp((v+27)/5)-1);
m_e_inf=alpha_m./(alpha_m+beta_m);
:minf = 0.32*(v+54)/(1-exp(-(v+54)/4))/(0.32*(v+54)/(1-exp(-(v+54)/4))+0.28*(v+27)/(exp((v+27)/5)-1))
:mtau = 1/(0.32*(v+54)/(1-exp(-(v+54)/4))+0.28*(v+27)/(exp((v+27)/5)-1))

function h_e_inf=h_e_inf(v)
alpha_h=0.128*exp(-(v+50)/18);
beta_h=4./(1+exp(-(v+27)/5));
h_e_inf=alpha_h./(alpha_h+beta_h);
:hinf = 0.128*exp(-(v+50)/18)/(0.128*exp(-(v+50)/18)+4/(1+exp(-(v+27)/5)))
:htau = 1/(0.128*exp(-(v+50)/18)+4/(1+exp(-(v+27)/5)))

Regression fit INF
minf = 1.0/(1.0+(exp((v+41.03)/(-7.41))))
hinf = 1.0/(1.0+(exp((v+45.32)/(4.04))))

Calculated TAU
mtau = (1 - exp(-v/4 - 27/2))*(1 - exp(v/5 + 27/5))/(-(1 - exp(-v/4 - 27/2))*(0.28*v + 7.56) + (1 - exp(v/5 + 27/5))*(0.32*v + 17.28))
htau = (exp(-v/5 - 27/5) + 1)/(0.128*(exp(-v/5 - 27/5) + 1)*exp(-v/18 - 25/9) + 4)

ENDCOMMENT

PROCEDURE rate(v (mV)) {
	UNITSOFF
	minf = 1.0/(1.0+(exp((v+minfvhalf)/(minfk))))
	:minf = 0.32*(v+54)/(1-exp(-(v+54)/4))/(0.32*(v+54)/(1-exp(-(v+54)/4))+0.28*(v+27)/(exp((v+27)/5)-1))
	mtau = (1 - exp(-v/4 - 27/2))*(1 - exp(v/5 + 27/5))/(-(1 - exp(-v/4 - 27/2))*(0.28*v + 7.56) + (1 - exp(v/5 + 27/5))*(0.32*v + 17.28))   
	:mtau = 1/(0.32*(v+54)/(1-exp(-(v+54)/4))+0.28*(v+27)/(exp((v+27)/5)-1))

	hinf = 1.0/(1.0+(exp((v+hinfvhalf)/(hinfk))))  
	:hinf = 0.128*exp(-(v+50)/18)/(0.128*exp(-(v+50)/18)+4/(1+exp(-(v+27)/5)))
	htau = (exp(-v/5 - 27/5) + 1)/(0.128*(exp(-v/5 - 27/5) + 1)*exp(-v/18 - 25/9) + 4)
	:htau = 1/(0.128*exp(-(v+50)/18)+4/(1+exp(-(v+27)/5)))
	UNITSON
}