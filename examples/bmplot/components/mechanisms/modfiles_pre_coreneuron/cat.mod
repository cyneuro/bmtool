:Transient Ca current channel (I_CaT)

NEURON {
        SUFFIX cat
	USEION ca READ eca WRITE ica
	RANGE gbar, g
	RANGE uinf, zinf, utau, ztau 
	RANGE ica
}

UNITS {
        (mA) = (milliamp)
	(mV) = (millivolt)
}

PARAMETER {
	gbar (siemens/cm2)
}

STATE { u z }

ASSIGNED {
	v (mV)
	eca (mV)
	ica (mA/cm2)
	uinf
	zinf 
	utau (ms)
	ztau (ms)
	g (siemens/cm2)
}

BREAKPOINT { 
	SOLVE states METHOD cnexp
	g = gbar*u*u*u*z
	ica = g*(v-eca)
}

INITIAL {
	rate(v)
	u = uinf
	z = zinf
}

DERIVATIVE states {
	rate(v)
	u' = (uinf-u)/utau
	z' = (zinf-z)/ztau
}

PROCEDURE rate(v(mV)) {
	UNITSOFF
	uinf = 1.0/(1.0+ (exp ((v+27.1)/(-7.2))))       
	utau = (43.4 - 42.6/(1.0+ (exp ((v+68.1)/(-20.5)))))         

	zinf = 1.0/(1.0+(exp ((v+32.1)/(5.5))))       
	ztau = (210 - 179.6/(1.0+(exp ((v+55)/(-16.9)))))   
	UNITSON	
}
