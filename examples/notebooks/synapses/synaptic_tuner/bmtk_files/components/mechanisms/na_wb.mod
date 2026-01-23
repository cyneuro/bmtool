COMMENT
Wang-Buzsaki Model of an Inhibitory
Interneuron in Rat Hippocampus

Sodium Channel

Reference: Borgers - An Introduction to Modeling Neuronal Dynamics Chapter 5
.mod by Tyler Banks
ENDCOMMENT

NEURON {
	SUFFIX na_wb
	USEION na READ ena WRITE ina
	RANGE gbar, g
	RANGE minf, hinf, mtau, htau
	RANGE ina
	RANGE minfvhalf,minfk,hinfvhalf,hinfk
	RANGE htauphi
}

UNITS {
	(mA) = (milliamp)
	(mV) = (millivolt)
}

PARAMETER {
	gbar (siemens/cm2)
	minfvhalf = 34.57
    minfk = -9.25 :-9.55
    hinfvhalf = 55.16
    hinfk = 7.07
	htauphi = 5
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
Matlab code
function m_i_inf=m_i_inf(v)
alpha_m=0.1*(v+35)./(1-exp(-(v+35)/10));
beta_m=4*exp(-(v+60)/18);
m_i_inf=alpha_m./(alpha_m+beta_m);
:minf = 0.1*(v+35)/(1-exp(-(v+35)/10))/(0.1*(v+35)/(1-exp(-(v+35)/10))+4*exp(-(v+60)/18))
:mtau = 1/(0.1*(v+35)/(1-exp(-(v+35)/10))+4*exp(-(v+60)/18))

function h_i_inf=h_i_inf(v)
alpha_h=0.07*exp(-(v+58)/20);
beta_h=1./(exp(-0.1*(v+28))+1);
h_i_inf=alpha_h./(alpha_h+beta_h);
:hinf = 0.07*exp(-(v+58)/20)/(0.07*exp(-(v+58)/20)+1/(exp(-0.1*(v+28))+1))
:htau = 1/(0.07*exp(-(v+58)/20)+1/(exp(-0.1*(v+28))+1))

Regression fit INF
minf = 1.0/(1.0+(exp((v+34.57)/(-9.55))))
hinf = 1.0/(1.0+(exp((v+55.16)/(7.07))))

Calculated TAU
mtau = (1 - exp(-v/10 - 7/2))/(0.1*v + 4*(1 - exp(-v/10 - 7/2))*exp(-v/18 - 10/3) + 3.5)
htau = (exp(0.1*v) + 0.0608)/(0.07*(exp(0.1*v) + 0.0608)*exp(-v/20 - 29/10) + exp(0.1*v))

ENDCOMMENT

PROCEDURE rate(v (mV)) {
	UNITSOFF
	minf = 1.0/(1.0+(exp((v+minfvhalf)/(minfk))))
	mtau = (1 - exp(-v/10 - 7/2))/(0.1*v + 4*(1 - exp(-v/10 - 7/2))*exp(-v/18 - 10/3) + 3.5)   
	:minf = 0.1*(v+35)/(1-exp(-(v+35)/10))/(0.1*(v+35)/(1-exp(-(v+35)/10))+4*exp(-(v+60)/18))
	:mtau = 1/(0.1*(v+35)/(1-exp(-(v+35)/10))+4*exp(-(v+60)/18))

	hinf = 1.0/(1.0+(exp((v+hinfvhalf)/(hinfk))))
	htau = (exp(0.1*v) + 0.0608)/(0.07*(exp(0.1*v) + 0.0608)*exp(-v/20 - 29/10) + exp(0.1*v))
	:hinf = 0.07*exp(-(v+58)/20)/(0.07*exp(-(v+58)/20)+1/(exp(-0.1*(v+28))+1))
	:htau = 1/(0.07*exp(-(v+58)/20)+1/(exp(-0.1*(v+28))+1))

	htau = htau/htauphi
	UNITSON
}