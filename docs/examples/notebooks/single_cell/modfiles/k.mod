: spike-generating sodium channel (Pyramid)

NEURON {
	SUFFIX k
	USEION k READ ko,ki WRITE ik
	RANGE gkbar, gk
	RANGE ninf, ntau
	RANGE nvhalf, nk
	RANGE ncbase, ncamp, nvpeak, nsigma
}

UNITS {
	(mA) = (milliamp)
	(mV) = (millivolt)
	FARADAY = (faraday) (coulomb)
	R = (k-mole) (joule/degC)

}

PARAMETER {
	
	
	celsius = 6.3
	gkbar = 0.036 (siemens/cm2) <0,1e9>
	
	nvhalf = -53 (mV)
	nk = 15 (mV)
	
	ncbase = 1.1 (ms)
	ncamp = 4.72(ms)
	nvpeak = -79 (mV)
	nsigma = 50 (mV)
	
}

ASSIGNED {
	v (mV)
	ko (mM)
	ki (mM)
	ek (mV)
	ik (mA/cm2)
	ninf
	ntau (ms)
	gk (siemens/cm2)
}

STATE {
	n
}

BREAKPOINT {
	SOLVE states METHOD cnexp
	gk = gkbar*n*n*n*n
	ek = (1e3)*(R*(celsius+273.15))/(FARADAY)*log(ko/ki)
	ik = gk*(v-ek)
}

INITIAL {
	rate(v)
	n = ninf
}

DERIVATIVE states {
	rate(v)
	n' = (ninf-n)/ntau
}

PROCEDURE rate(v (mV)) {
	LOCAL na, nb
	UNITSOFF

	ninf = 1/(1+exp((nvhalf-v)/nk))
	ntau = ncbase+ncamp*exp((-(nvpeak-v)^2)/nsigma^2)

	UNITSON
}
