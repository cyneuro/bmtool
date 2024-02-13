TITLE Sodium persistent current for RD Traub, J Neurophysiol 89:909-921, 2003

COMMENT

	Implemented by Maciej Lazarewicz 2003 (mlazarew@seas.upenn.edu)

ENDCOMMENT

INDEPENDENT { t FROM 0 TO 1 WITH 1 (ms) }

UNITS { 
	(mV) = (millivolt) 
	(mA) = (milliamp) 
} 
NEURON { 
	SUFFIX nap
	USEION na READ ena WRITE ina
	RANGE i, minf, mtau, gnap, gbar :, vhalf, k
}

PARAMETER { 
	gbar = 1e-4 	(mho/cm2)
	v ena 		(mV)  
	k = 5      (mV)
	vhalf = -48 (mV)
} 
ASSIGNED { 
	ina 		(mA/cm2) 
	i   		(mA/cm2)
	minf 		(1)
	mtau 		(ms) 
	gnap		(mho/cm2)
} 
STATE {
	m
}

BREAKPOINT { 
	SOLVE states METHOD cnexp
	gnap = gbar * m
	ina = gnap * ( v - ena ) 
	i = ina
} 

INITIAL { 
	rate(v)
	m = minf
} 

DERIVATIVE states { 
	rate(v)
	m' = ( minf - m ) / mtau 
}
UNITSOFF
 
PROCEDURE rate(v (mV)) {
	if (v < -67.5 ) {
	minf = 0
	} else{
	minf  = 1 / ( 1 + exp( ( vhalf - v ) / k ) )
	}
	if( v < -40.0 ) {
		mtau = 100*(0.025 + 0.14 * exp( ( v + 40 ) / 10 ))
	}else{
		mtau = 100*(0.02 + 0.145 * exp( ( - v - 40 ) / 10 ))
	}
}
UNITSON
