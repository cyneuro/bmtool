: Two Ca2+ pools for ic and isAHP

NEURON {
    SUFFIX capool
	USEION ca READ ica
	USEION cas READ casi WRITE casi VALENCE 2 
	RANGE fcas, taucas, cainf,xx
}

UNITS {
        (mM) = (milli/liter)
        (mA) = (milliamp)
	(mV) = (millivolt)
	FARADAY = 96485.309 (coul)
}

PARAMETER {
	pi = 3.14159265
	taucas= 1000 (ms) 	: decay time constant
    cainf= 50e-6   (mM)  	: equilibrium ca2+ concentration
	fcas = 0.024
    w = 1 (micrometer)     	: thickness of shell for ca2+ diffusion
	z = 2			: valence
}

ASSIGNED {
	v (mV)
	ica (mA/cm2)
    A       (mM-cm2/ms/mA)
}

STATE { casi(mM) xx}

BREAKPOINT { 
	SOLVE states METHOD cnexp
}

INITIAL {
	A = 1/(z*FARADAY*w)*(1e4)
	casi = cainf
}

DERIVATIVE states {
	casi'= -fcas*A*ica + (cainf - casi)/taucas
	xx=casi
}



