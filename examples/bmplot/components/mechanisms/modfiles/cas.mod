: Persistent Ca2+ current channel(I_CaS)

NEURON {
	SUFFIX cass
	USEION ca READ eca WRITE ica
	RANGE gbar, g
	RANGE jainf, kinf, jatau, ktau
	RANGE ica
}

UNITS {
	(mA) = (milliamp)
	(mV) = (millivolt)
}

PARAMETER {
	gbar (siemens/cm2)
}

ASSIGNED {
	v (mV)
	eca (mV)
	ica (mA/cm2)
	jainf
	kinf 
	jatau (ms)
	ktau (ms)
	g (siemens/cm2)
}

STATE {
	ja k
}

BREAKPOINT {
	SOLVE states METHOD cnexp
	g = gbar*ja*ja*ja*k
	ica = g*(v-eca)
}

INITIAL {
	rate(v)
	ja = jainf
	k = kinf
}

DERIVATIVE states {
	rate(v)
	ja' = (jainf-ja)/jatau
	k' = (kinf-k)/ktau
}

PROCEDURE rate(v(mV)) {
	UNITSOFF
	jainf = (1.0)/(1+ (exp ((v+33.0)/(-8.1)))) 
	jatau = 2.8 + 14.0/( (exp ((v+27.0)/(10.0))) + (exp ((v+70.0)/(-13.0))))
	kinf = 1.0/(1.0+(exp ((v+60.0)/(6.2))))
	ktau = (120.0 + 300.0/( (exp ((v+55.0)/(9.0))) + (exp ((v+65)/(-16.0)))))
	UNITSON
}
	

