: passive leak current

NEURON {
	SUFFIX leakinter
	NONSPECIFIC_CURRENT il
	RANGE il, el, glbar_inter
}

UNITS {
	(mA) = (milliamp)
	(mV) = (millivolt)
}

PARAMETER {
	glbar_inter = 3.333333e-5 (siemens/cm2) < 0, 1e9 >
	el = -75 (mV)
}

ASSIGNED {
	v (mV)
	il (mA/cm2)
}

BREAKPOINT { 
	il = glbar_inter*(v - el)
}
