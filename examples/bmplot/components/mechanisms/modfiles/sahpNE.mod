:  iC   fast Ca2+/V-dependent K+ channel

NEURON {
	SUFFIX sAHPNE
	USEION k READ ek WRITE ik
	USEION cas READ casi VALENCE 2 
        RANGE ik, gk, gsAHPbar
}

UNITS {
        (mM) = (milli/liter)
	(mA) = (milliamp)
	(mV) = (millivolt)
}

PARAMETER {
	tone_period = 4000  
	NE_period = 500
	NE_start = 64000 : 36000		   : NE beta-R(Low Affinity) Norepinephrine Effect after 1 conditioning trials (9*4000 = 36000)
	NE_stop = 96000
	NE_t1 = 0.9 : 0.9           : Amount(%) of NE effect
	NE_ext1 = 196000
	NE_ext2 = 212000	
	
	NE_period2 = 100
	NE_start2 = 36000		   : NE beta-R(Low Affinity) Norepinephrine Effect after 0 conditioning trials (8*4000 = 32000)
	NE_t2 = 0.7           : Amount(%) of NE effect	
	
	gsAHPbar= 2.318144e-05 : 0.0001	(mho/cm2) : 
}

ASSIGNED {
	v (mV)
	ek (mV)
	casi (mM)
	ik (mA/cm2)
	cinf 
	ctau (ms)
	gk (mho/cm2)
}

STATE {
	c
}

BREAKPOINT {
	SOLVE states METHOD cnexp
	gk = gsAHPbar*c       
	ik = gk*(v-ek)*NE1(t)*NE2(t)
}

INITIAL {
	rate(v,casi)
	c = cinf
}

DERIVATIVE states {
        rate(v,casi)
	c' = (cinf-c)/ctau
}

UNITSOFF


FUNCTION calf(v (mV), casi (mM)) (/ms) { LOCAL vs, va
	UNITSOFF
	vs=10*log10(1000*casi)
	calf = 0.0048/exp(-0.5*(vs-35))
	UNITSON
}

FUNCTION cbet(v (mV), casi (mM))(/ms) { LOCAL vs, vb 
	UNITSOFF
	  vs=10*log10(1000*casi)
	  cbet = 0.012/exp(0.2*(vs+100))
	UNITSON
}

UNITSON

PROCEDURE rate(v (mV), casi (mM)) {LOCAL  csum, ca, cb
	UNITSOFF
	ca=calf(v, casi) 
	cb=cbet(v, casi)		
	csum = ca+cb
    if (v < -65 ) {              :::::::::::::::::::::::::::  67.5
	cinf = 0
	} else{
	cinf = ca/csum
	}
	ctau = 48
	UNITSON
}

FUNCTION NE1(t) {
	    if (t >= NE_start && t <= NE_stop){ 									: During conditioning
			if ((t/tone_period-floor(t/tone_period)) >= (1-NE_period/tone_period)) {NE1 = NE_t1}
			else if ((t/tone_period-floor(t/tone_period)) == 0) {NE1 = NE_t1}
			else {NE1 = 1}}
		else if (t >= NE_ext1 && t <= NE_ext2){								    : During 4trials of Extinction
			if ((t/tone_period-floor(t/tone_period)) >= (1-NE_period/tone_period)) {NE1 = NE_t1}
			else if ((t/tone_period-floor(t/tone_period)) == 0) {NE1 = NE_t1}
			else {NE1 = 1}}		
		else  {NE1 = 1}
	}
FUNCTION NE2(t) {
	    if (t >= NE_start2 && t <= NE_stop){
			if((t/tone_period-floor(t/tone_period)) >= (1-NE_period2/tone_period)) {NE2 = NE_t2}
			else if ((t/tone_period-floor(t/tone_period)) == 0) {NE2 = NE_t2}
			else  {NE2 = 1}}
		else  {NE2 = 1}
	}	