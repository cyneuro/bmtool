: passive leak current

NEURON {
	SUFFIX leakCA3
	NONSPECIFIC_CURRENT il
	RANGE il, el, glbar
}

UNITS {
	(mA) = (milliamp)
	(mV) = (millivolt)
}

PARAMETER {
	glbar = 6e-05  :3.333333e-5 (siemens/cm2) < 0, 1e9 >
	el = -60 (mV)
}

ASSIGNED {
	v (mV)
	il (mA/cm2)
}

BREAKPOINT { 
	il = glbar*(v - el)
}	