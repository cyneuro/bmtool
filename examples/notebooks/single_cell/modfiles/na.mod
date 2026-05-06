: spike-generating sodium channel (Pyramid)

NEURON {
	SUFFIX na
	USEION na READ nao,nai WRITE ina
	RANGE gnabar, gna
	RANGE minf, hinf, mtau, htau
	RANGE mvhalf, hvhalf, mk, hk
	RANGE mcbase, hcbase, mcamp, hcamp, mvpeak, hvpeak, msigma, hsigma
}

UNITS {
	(mA) = (milliamp)
	(mV) = (millivolt)
	FARADAY = (faraday) (coulomb)
	R = (k-mole) (joule/degC)

}

PARAMETER {
	
	celsius = 6.3
	gnabar = 0.12 (siemens/cm2) <0,1e9>
	
	mvhalf = -40 (mV)
	mk = 10 (mV)
	
	mcbase = 0.04 (ms)
	mcamp = 0.092 (ms)
	mvpeak = -38 (mV)
	msigma = 30 (mV)
	
	hvhalf = -69 (mV)
	hk = -3.5 (mV)
	
	hcbase = 1.2 (ms)
	hcamp = 7.4 (ms)
	hvpeak = -67 (mV)
	hsigma = 20 (mV)
	
}

ASSIGNED {
	v (mV)
	nao (mM)
	nai (mM)
	ena (mV)
	ina (mA/cm2)
	minf
	hinf
	mtau (ms)
	htau (ms)
	gna (siemens/cm2)
}

STATE {
	m h
}

BREAKPOINT {
	SOLVE states METHOD cnexp
	gna = gnabar*m*m*m*h
	ena = (1e3)*(R*(celsius+273.15))/(FARADAY)*log(nao/nai)
	ina = gna*(v-ena)
}

INITIAL {
	rate(v)
	m = 0.09
	h = 0.7
}

DERIVATIVE states {
	rate(v)
	m' = (minf-m)/mtau
	h' = (hinf-h)/htau
}

PROCEDURE rate(v (mV)) {
	
	minf = 1/(1+exp((mvhalf-v)/mk))
	mtau = mcbase+mcamp*exp((-(mvpeak-v)^2)/msigma^2)
	
	hinf = 1/(1+exp((hvhalf-v)/hk))
	htau = hcbase+hcamp*exp((-(hvpeak-v)^2)/hsigma^2)
	
}
