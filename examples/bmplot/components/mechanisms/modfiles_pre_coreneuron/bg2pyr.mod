:Background to Pyramidal Cells AMPA+NMDA 

NEURON {
	POINT_PROCESS bg2pyr
	NONSPECIFIC_CURRENT inmda
	NONSPECIFIC_CURRENT iampa
	RANGE taun1, taun2, factorn, normconstn
	RANGE taua1, taua2, factora, normconsta
	RANGE gnmda, gnmdas, gNMDAmax, enmda
	RANGE gampa, gampas, gAMPAmax, eampa
	RANGE initW
}

UNITS {
	(mV) = (millivolt)
        (nA) = (nanoamp)
	(uS) = (microsiemens)
}

PARAMETER {

	:W
	initW = 6.3 : 6.3 : 6.3 : 8 :6

	:NMDA
	taun1 = 5 (ms)
	taun2 = 125 (ms)
	gNMDAmax = 0.5e-3 (uS)
	enmda = 0 (mV)

	:AMPA
	taua1 = .5 (ms)
	taua2 = 7 (ms)
	gAMPAmax = 1e-3 (uS)
	eampa = 0 (mV)
	
}

ASSIGNED {
	v (mV)
	eca (mV)
	
	:NMDA
	inmda (nA)
	gnmda
	gnmdas
	factorn
	normconstn

	:AMPA
	iampa (nA)
	gampa
	gampas
	factora
	normconsta
}

STATE {

	:NMDA
	An
	Bn

	:AMPA
	Aa
	Ba
}

INITIAL {

	:NMDA
	An = 0
	Bn = 0
	factorn = taun1*taun2/(taun2-taun1)
	normconstn = -1/(factorn*(1/exp(log(taun2/taun1)/(taun1*(1/taun1-1/taun2)))-1/exp(log(taun2/taun1)/(taun2*(1/taun1-1/taun2)))))

	:AMPA
	Aa = 0
	Ba = 0
	factora = taua1*taua2/(taua2-taua1)
	normconsta = -1/(factora*(1/exp(log(taua2/taua1)/(taua1*(1/taua1-1/taua2)))-1/exp(log(taua2/taua1)/(taua2*(1/taua1-1/taua2)))))
}

BREAKPOINT {
	SOLVE states METHOD cnexp
	gnmda = normconstn*factorn*(Bn-An)
	gnmdas = gnmda
	if (gnmdas>1) {gnmdas=1}
	inmda = initW*gNMDAmax*gnmdas*(v-enmda)*sfunc(v)
	
	gampa = normconsta*factora*(Ba-Aa)
	gampas = gampa
	if (gampas > 1) {gampas = 1}
	iampa = initW*gAMPAmax*gampas*(v-eampa)
	
}

DERIVATIVE states {

	:NMDA
	An' = -An/taun1
	Bn' = -Bn/taun2
	

	:AMPA
	Aa' = -Aa/taua1
	Ba' = -Ba/taua2

}

NET_RECEIVE(wgt) {
        LOCAL x
	x = wgt
	state_discontinuity(An,An+x)
	state_discontinuity(Bn,Bn+x)
	state_discontinuity(Aa,Aa+x)
	state_discontinuity(Ba,Ba+x)
}

:::::::::::: FUNCTIONs and PROCEDUREs ::::::::::::
FUNCTION sfunc (v (mV)) {
	UNITSOFF
	sfunc = 1/(1+0.33*exp(-0.06*v))
	UNITSON
}
