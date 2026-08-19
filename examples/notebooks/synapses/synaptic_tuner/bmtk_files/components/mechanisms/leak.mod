: passive leak current

NEURON {
	SUFFIX leak
	USEION leak READ eleak WRITE ileak VALENCE 1
	RANGE ileak, eleak, gbar
}

UNITS {
	(mA) = (milliamp)
	(mV) = (millivolt)
}

PARAMETER {
	gbar (siemens/cm2)
	eleak (mV)
}

ASSIGNED {
	v (mV)
	ileak (mA/cm2)
}

BREAKPOINT { 
	ileak = gbar*(v - eleak)
}