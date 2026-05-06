: passive leak current

NEURON {
	SUFFIX leak
	NONSPECIFIC_CURRENT il
	RANGE il, el, glbar
}

UNITS {
	(mA) = (milliamp)
	(mV) = (millivolt)
}

PARAMETER {
	glbar = 2.857142857142857e-05  :3.333333e-5 (siemens/cm2) < 0, 1e9 >
	el = -75 (mV)
}

ASSIGNED {
	v (mV)
	il (mA/cm2)
}

BREAKPOINT { 
	il = glbar*(v - el)
}	