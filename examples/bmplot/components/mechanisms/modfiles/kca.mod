:  ca-dependent potassium current

NEURON {
	SUFFIX kca
	USEION k READ ek WRITE ik
	USEION ca READ ica
        RANGE g, gbar, ik
}

UNITS {
        (mM) = (milli/liter)
	(mA) = (milliamp)
	(mV) = (millivolt)
}

PARAMETER {
	gbar (siemens/cm2)
}

ASSIGNED {
	v (mV)
	ek (mV)
	ik (mA/cm2)
	cinf 
	ctau (ms)
	g (siemens/cm2)
	ica (mM)
}

STATE {
	c
}


BREAKPOINT {
	SOLVE states METHOD cnexp
	g = gbar*c*c*c*c       
	ik = g*(v-ek)
}


INITIAL {
	rate(v,ica)
	c = cinf
}

DERIVATIVE states {
        rate(v,ica)
	c' = (cinf-c)/ctau
}

PROCEDURE rate(v (mV), ica (mM)) {
	UNITSOFF
	:activation based on internal concentration of capool
	cinf = ((ica)/(ica + 0.003))*((1.0)/(1+ (exp (-(v+28.3)/(12.6)))))       :/(ica + 0.003)) 
	ctau = ((180.6)-(150.2)/(1+(exp (-(v+46)/(22.7))))) 
	UNITSON		
}


